import collections
import warnings
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from vad.configs.train_config import TrainConfig
from vad.losses import TokenNLLLoss
from vad.metrics import accuracy_metric_tensor, equal_error_rate, vad_accuracy
from vad.training.runner import Runner
from vad.training.training_info import TrainingInfo


class ModelRunner(Runner):
    config: TrainConfig
    context_window_frames: int

    def __init__(self, config: TrainConfig, context_window_frames: int):
        super().__init__()

        self.config = config
        self.context_window_frames = context_window_frames
        self.loss_function = TokenNLLLoss()

    def training_step(self, model: nn.Module, batch: Dict, info: TrainingInfo):
        inputs, targets = batch

        outputs = model(inputs["feature"])

        loss = self.loss_function(outputs, targets)

        predictions = outputs.argmax(dim=-1)
        accuracy = accuracy_metric_tensor(targets, predictions)

        return {
            "loss": loss.mean(),
            "acc": accuracy,
        }

    def training_step(self, model: nn.Module, batch: Dict, info: TrainingInfo):
        inputs, targets = batch
        outputs = model(inputs["feature"])

        loss = self.loss_function(outputs, targets)

        predictions = outputs.argmax(dim=-1)
        accuracy = accuracy_metric_tensor(targets, predictions)

        probabilities = torch.nn.functional.softmax(outputs, dim=-1).view(-1, 2)[:, 1]
        flat_targets = targets.view(-1).detach().cpu().numpy()
        flat_probabilities = probabilities.detach().cpu().numpy()

        try:
            train_auc = roc_auc_score(flat_targets, flat_probabilities)
        except ValueError:
            train_auc = 0.0

        return {
            "loss": loss.mean(),
            "acc": accuracy,
            "train_auc": train_auc,
        }

    def validation_step(self, model: torch.nn.Module, batch: Dict, info: TrainingInfo):
        inputs, targets = batch

        outputs = model(inputs["feature"])

        loss = self.loss_function(outputs, targets)

        predictions = outputs.argmax(dim=-1)
        accuracy = accuracy_metric_tensor(targets, predictions)

        probabilities = torch.nn.functional.softmax(outputs, dim=-1).view(-1, 2)[:, 1]

        loss = loss.unsqueeze(dim=0)
        accuracy = accuracy.unsqueeze(dim=0).to(device=loss.device)

        return {
            "val_loss": loss,
            "val_acc": accuracy,
            "probabilities": probabilities,
            "outputs": outputs,
            "positions": inputs["positions"],
            "data-index": inputs["data-index"],
            "data-length": inputs["data-length"],
            "labels": targets,
        }

    def validation_epoch_end(self, val_results: Dict[str, List[Any]]):
        # Not exact calculation due to batch size mismatch, but can reduce memory consumption
        val_loss = np.mean(val_results["val_loss"])
        val_accuracy = np.mean(val_results["val_acc"])

        labels = np.concatenate([result.flatten() for result in val_results["labels"]])
        probabilities = np.concatenate(
            [result.flatten() for result in val_results["probabilities"]]
        )

        if len(np.unique(labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(labels, probabilities)

        threshold = 0.5
        pred_labels = (probabilities > threshold).astype(np.int32)
        true_labels = labels.astype(np.int32)

        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        result = {
            "val_auc": auc,
            "val_accuracy": val_accuracy,
            "val_acc": val_accuracy,
            "val_loss": val_loss,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }

        val_data_lengths = {}
        for data_indices, data_lengths in zip(
            val_results["data-index"], val_results["data-length"]
        ):
            for data_index, data_length in zip(data_indices, data_lengths):
                val_data_lengths[int(data_index)] = int(data_length)

        if self.config.model.name in ("bdnn", "acam", "self-attention"):

            boosted_metrics = collections.defaultdict(list)
            for i, (data_index, data_length) in enumerate(val_data_lengths.items()):
                label_length = (
                    data_length + 2 * self.config.context_resolution.context_window_half_frames
                )
                boosted_outputs = np.zeros(
                    shape=(label_length, self.context_window_frames, 2), dtype=np.float32
                )
                boosted_counts = np.zeros(
                    shape=(label_length, self.context_window_frames, 1), dtype=np.float32
                )
                total_labels = np.zeros(label_length, dtype=np.float32)

                for i, val_data_index in enumerate(val_results["data-index"]):
                    indexed_outputs = val_data_index == data_index
                    if not indexed_outputs.any():
                        continue

                    outputs_array = val_results["outputs"][i][indexed_outputs]
                    positions_array = val_results["positions"][i][indexed_outputs]
                    labels_array = val_results["labels"][i][indexed_outputs]

                    window_indices = np.arange(self.context_window_frames)
                    window_indices = np.expand_dims(window_indices, axis=0).repeat(
                        len(positions_array), axis=0
                    )

                    boosted_outputs[positions_array, window_indices] = outputs_array
                    boosted_counts[positions_array, window_indices] = 1
                    total_labels[positions_array] = labels_array

                boosted_average = boosted_outputs.sum(axis=1) / (
                    boosted_counts.sum(axis=1) + np.finfo(np.float32).eps
                )

                boosted_probabilities = softmax(boosted_average, axis=1)
                boosted_predictions = boosted_average.argmax(axis=-1)

                vacc, acc, sba, eba, bp = vad_accuracy(total_labels, boosted_predictions)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="No positive samples in y_true, true positive value should be meaningless",
                    )

                    labels_np = np.asarray(total_labels).astype(np.int64)

                    if i < 3:
                        n_pos = int((labels_np == 1).sum())
                        n_neg = int((labels_np == 0).sum())
                        print(f"[val] item={i} label counts: pos={n_pos} neg={n_neg} total={labels_np.size}", flush=True)
                        print(f"[val] item={i} unique labels: {np.unique(labels_np)}", flush=True)

                    eer = equal_error_rate(total_labels, boosted_predictions)

                boosted_metrics["vacc"].append(vacc)
                boosted_metrics["sba"].append(sba)
                boosted_metrics["eba"].append(eba)
                boosted_metrics["bp"].append(bp)
                boosted_metrics["eer"].append(eer)

                if len(np.unique(total_labels)) < 2:
                    auc = 0.0
                else:
                    auc = roc_auc_score(total_labels, boosted_probabilities[:, 1])

                boosted_metrics["auc"].append(auc)

            result["boosted_val_auc"] = np.mean(boosted_metrics["auc"])
            result["boosted_val_vacc"] = np.mean(boosted_metrics["vacc"])
            result["boosted_val_sba"] = np.mean(boosted_metrics["sba"])
            result["boosted_val_eba"] = np.mean(boosted_metrics["eba"])
            result["boosted_val_bp"] = np.mean(boosted_metrics["bp"])
            result["boosted_val_eer"] = np.mean(boosted_metrics["eer"])

        return result