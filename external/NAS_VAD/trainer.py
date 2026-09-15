import numpy as np
import os
import random
import sys 
import time
import torch
import torch.nn as nn
import torchaudio
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
import warnings
from glob import glob
import torch.nn.functional as Fs
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import json
sys.path.append('../')
warnings.filterwarnings('ignore')

from darts.cnn.acam import *
from darts.cnn.genotypes import Genotype
from darts.cnn.model import *
from darts.cnn.sl_model import *
from darts.cnn.utils import count_parameters_in_MB, save, AvgrageMeter, accuracy, Cutout
from darts.darts_config import *
from misc.random_string import random_generator

# VOiCES metrics heper functions
def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


def compute_binary_metrics(preds, targets, threshold=0.5):
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    pred_labels = (preds >= threshold).astype(int)
    true_labels = np.round(targets).astype(int)

    auc = safe_auc(true_labels, preds)
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    f2 = fbeta_score(true_labels, pred_labels, beta=2, zero_division=0)

    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (true_labels == 0)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())

    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "auc": auc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.clamp(preds, min=1e-7, max=1.0 - 1e-7)

        bce = F.binary_cross_entropy(
            preds,
            targets,
            reduction="none",
        )

        pt = torch.where(targets >= 0.5, preds, 1.0 - preds)

        focal_weight = (1.0 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, negative_weight, positive_weight):
        super().__init__()
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight

    def forward(self, preds, targets):
        weights = torch.where(
            targets >= 0.5,
            torch.full_like(targets, self.positive_weight),
            torch.full_like(targets, self.negative_weight),
        )

        return F.binary_cross_entropy(
            preds,
            targets,
            weight=weights,
        )


def compute_balanced_class_weights(label_tensors):
    positive = 0
    total = 0

    for labels in label_tensors:
        labels = labels.reshape(-1)
        positive += int((labels >= 0.5).sum().item())
        total += labels.numel()

    negative = total - positive

    if positive == 0 or negative == 0:
        raise ValueError("weighted BCE requires both positive and negative training labels")

    positive_weight = total/(2.0 * positive)
    negative_weight = total/(2.0 * negative)

    return negative_weight, positive_weight


def build_loss(loss_type, train_label_tensors):
    if loss_type == "bce":
        return nn.BCELoss()

    if loss_type == "weighted_bce":
        negative_weight, positive_weight = compute_balanced_class_weights(train_label_tensors)

        print(
            f"Weighted BCE | "
            f"negative_weight={negative_weight:.6f} | "
            f"positive_weight={positive_weight:.6f}"
        )

        return WeightedBCELoss(
            negative_weight=negative_weight,
            positive_weight=positive_weight,
        )

    if loss_type == "focal":
        return BinaryFocalLoss(
            gamma=2.0,
        )

    raise ValueError(
        f"Unsupported loss type: {loss_type}"
    )


def parse_voices_room_and_noise(filepath):
    name = os.path.basename(filepath)

    # noise type
    if "-tele-" in name:
        noise = "tele"
    elif "-babb-" in name:
        noise = "babb"
    else:
        noise = "unknown"

    # room
    room = "unknown"
    parts = name.split("-")
    for part in parts:
        if part.startswith("rm"):
            room = part
            break

    return room, noise

# Voicebank metrics helper functions
def load_voicebank_map(txt_path):
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            utt_id = parts[0]
            noise = parts[1]
            try:
                snr = float(parts[2])
            except ValueError:
                continue
            mapping[utt_id] = {"noise": noise, "snr": snr}
    return mapping


def parse_voicebank_utt_id(filepath):
    name = os.path.basename(filepath)
    name = name.replace("_spec.npy", "").replace(".npy", "")
    return name


def format_snr_key(snr):
    # makes keys like snr_17.5, snr_12.5, etc.
    return f"snr_{snr:g}"

def read_manifest(jsonl_path):
    spec_files = []
    label_files = []
    with open(jsonl_path, "r") as f:
        for line in f:
            row = json.loads(line)
            spec_files.append(row["spec_path"])
            label_files.append(row["label_path"])
    return spec_files, label_files

# ONDRI
def parse_ondri_metadata(filepath):
    cohort = os.path.basename(os.path.dirname(filepath))

    name = os.path.basename(filepath)
    name = name.replace("_spec.npy", "").replace(".npy", "")

    parts = name.split("_")

    study_code = parts[0] if len(parts) > 0 else "unknown"
    site_id = parts[1] if len(parts) > 1 else "unknown"
    participant_id = parts[2] if len(parts) > 2 else "unknown"
    timepoint = parts[3] if len(parts) > 3 else "unknown"

    cohort_map = {
        "1": "ADMCI",
        "2": "ALS",
        "3": "FTD",
        "4": "PD",
        "5": "VCI",
    }

    if participant_id != "unknown" and participant_id:
        cohort = cohort_map.get(participant_id[0], "unknown")
    else:
        cohort = "unknown"

    raw_activity = "unknown"

    if len(parts) > 4:
        raw_activity = parts[4].split("-")[0]

    activity = raw_activity.lower()

    if activity in {"puh", "tuh", "kuh"}:
        task_type = "AMR"
    elif activity in {"puhtuhkuh", "buttercup"}:
        task_type = "SMR"
    elif activity in {"seqstort1", "seqstort2a"}:
        task_type = "NARRATIVE"
    else:
        task_type = "unknown"

    return {
        "cohort": cohort,
        "study_code": study_code,
        "site_id": site_id,
        "participant_id": participant_id,
        "timepoint": timepoint,
        "activity": activity,
        "task_type": task_type,
    }

def add_to_metric_bucket(buckets, key, pred_np, target_np):
    if key not in buckets:
        buckets[key] = {
            "preds": [],
            "targets": [],
            "n_files": 0,
        }

    buckets[key]["preds"].append(pred_np)
    buckets[key]["targets"].append(target_np)
    buckets[key]["n_files"] += 1


def print_metric_bucket(label, bucket):
    bucket_preds = np.concatenate(bucket["preds"], axis=0)
    bucket_targets = np.concatenate(bucket["targets"], axis=0)

    metrics = compute_binary_metrics(bucket_preds, bucket_targets)

    print(
        f"{label} | files={bucket['n_files']} | "
        f"auc={metrics['auc']:.4f} | "
        f"acc={metrics['acc']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"f2={metrics['f2']:.4f} | "
        f"miss_rate={metrics['miss_rate']:.4f} | "
        f"false_alarm_rate={metrics['false_alarm_rate']:.4f} | "
        f"TP={metrics['tp']} | TN={metrics['tn']} | "
        f"FP={metrics['fp']} | FN={metrics['fn']}"
    )

class Trainer:
    def __init__(self,
                 data_path,
                 model_save_path: str,
                 dataset: str = 'cv7',
                 epochs=50,
                 mode='train',
                 model=None,
                 model_type='NewSearch',
                 test_dataset = 'None',
                 window=[-19, -9, -1, 0, 1, 9, 19],
                 n_mels=80,
                 batch_size=128,
                 learning_rate=1e-3,
                 weight_decay=0.0,
                 loss_type="bce",
                 patience=10,
                 grad_clip=GRAD_CLIP,
                 drop_path=0.0,
                 ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.patience = patience
        self.grad_clip = grad_clip
        self.drop_path = drop_path
        self.mode = mode
        self.model = model
        self.model_type = model_type
        self.epochs = epochs
        self.dataset_name = dataset
        self.save_path = model_save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.window = window
        self.test_data = test_dataset

        min_size = 700
        train_path, valid_path = self.data_path.split(',')

        if self.mode == 'train':
            if train_path.endswith(".jsonl"):
                train_files, train_label_files = read_manifest(train_path)
            else:
                train_label_files = sorted([
                    os.path.join(train_path, f)
                    for f in os.listdir(train_path)
                    if f.endswith('.npy') and 'spec' not in f
                    and os.stat(os.path.join(train_path, f)).st_size > min_size
                ])
                random.shuffle(train_label_files)
                train_files = [item.replace('.npy', '_spec.npy') for item in train_label_files]

            train_dataset = VAD_Dataset(
                train_files,
                train_label_files,
                n_fft=400,
                n_mels=n_mels,
                sample_rate=16000,
                mode=self.mode,
                model_type=self.model_type
            )
            print(len(train_label_files))

        if valid_path.endswith(".jsonl"):
            valid_files, valid_label_files = read_manifest(valid_path)
        else:
            if self.mode == 'test' and self.dataset_name in {'ONDRI-DDK','ONDRI-NARRATIVE'}:
                valid_label_files = sorted([
                    p for p in glob(os.path.join(valid_path, '**', '*.npy'), recursive=True)
                    if 'spec' not in os.path.basename(p)
                    and os.stat(p).st_size > min_size
                ])
            else:
                valid_label_files = sorted([
                    os.path.join(valid_path, f)
                    for f in os.listdir(valid_path)
                    if f.endswith('.npy') and 'spec' not in f
                    and os.stat(os.path.join(valid_path, f)).st_size > min_size
                ])

            valid_files = [item.replace('.npy', '_spec.npy') for item in valid_label_files]

        if self.mode == 'train':
            valid_dataset = VAD_Dataset(
                valid_files,
                valid_label_files,
                n_fft=400,
                n_mels=n_mels,
                sample_rate=16000,
                model_type=self.model_type,
                mode='valid'
            )
        else:
            valid_dataset = VAD_Dataset(
                valid_files,
                valid_label_files,
                n_fft=400,
                n_mels=n_mels,
                sample_rate=16000,
                model_type=self.model_type,
                mode='test'
            )         
        
        if self.mode =='train':
            self.train_data = train_dataset
        self.valid_data = valid_dataset

    def train(self):
        criterion = build_loss(
            self.loss_type,
            self.train_data.label_files,
        ).cuda()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
            eta_min=1e-6,
        )

        train_queue = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=0,
            shuffle=True,
        )

        valid_queue = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=0,
        )

        early = 0
        best_valid_f2 = -np.inf
        best_checkpoint = None
        best_epoch = None
        best_metrics = None

        print(f"Starting training for {self.epochs} epochs")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Valid samples: {len(self.valid_data)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Loss: {self.loss_type}")
        print(f"Early-stop patience: {self.patience}")
        print(f"Gradient clip: {self.grad_clip}")
        print(f"Model: {self.model_type}")
        print(f"Dataset: {self.dataset_name}")

        if self.model_type == "NewSearch":
            print(f"Drop path: {self.drop_path}")

        for e in range(self.epochs):
            print(f"\nEpoch {e+1}/{self.epochs}")

            if hasattr(self.model, "drop_path_prob"):
                self.model.drop_path_prob = self.drop_path

            start = time.time()

            train_auc, train_obj = train_step(
                train_queue,
                self.model,
                criterion,
                optimizer,
                self.model_type,
                self.grad_clip,
            )

            if e == 0:
                print(
                    f"# params: "
                    f"{sum(p.numel() for p in self.model.parameters())}"
                )

            (
                valid_auc,
                valid_acc,
                valid_precision,
                valid_recall,
                valid_f1,
                valid_f2,
                valid_obj,
            ) = valid_step(
                valid_queue,
                self.model,
                criterion,
                self.dataset_name,
                self.model_type,
            )

            scheduler.step()

            print(
                f"Epoch {e+1}/{self.epochs} | "
                f"train_auc={train_auc:.4f} | "
                f"train_loss={train_obj:.6f} | "
                f"valid_auc={valid_auc:.4f} | "
                f"valid_acc={valid_acc:.4f} | "
                f"valid_precision={valid_precision:.4f} | "
                f"valid_recall={valid_recall:.4f} | "
                f"valid_f1={valid_f1:.4f} | "
                f"valid_f2={valid_f2:.4f} | "
                f"valid_loss={valid_obj:.6f}"
            )

            if valid_f2 > best_valid_f2:
                best_valid_f2 = valid_f2
                best_epoch = e + 1

                new_checkpoint = (
                    f"{self.save_path}/"
                    f"{e:03d}_{self.model_type}_{self.dataset_name}.pth"
                )

                # Keep only the current best checkpoint.
                if (
                    best_checkpoint is not None
                    and os.path.isfile(best_checkpoint)
                ):
                    os.remove(best_checkpoint)

                torch.save(
                    self.model.state_dict(),
                    new_checkpoint,
                )

                best_checkpoint = new_checkpoint

                best_metrics = {
                    "auc": float(valid_auc),
                    "acc": float(valid_acc),
                    "precision": float(valid_precision),
                    "recall": float(valid_recall),
                    "f1": float(valid_f1),
                    "f2": float(valid_f2),
                    "loss": float(valid_obj),
                }

                print(
                    f"Saved new best model at epoch {e+1} "
                    f"with valid_f2={valid_f2:.6f}"
                )

                early = 0

            else:
                early += 1

            if early >= self.patience:
                print(
                    f"Early stopping at epoch {e+1} | "
                    f"best_valid_f2={best_valid_f2:.6f}"
                )
                break

        return {
            "best_checkpoint": best_checkpoint,
            "best_epoch": best_epoch,
            "best_valid_f2": float(best_valid_f2),
            "best_valid_metrics": best_metrics,
        }

    def test(self):
        criterion = nn.BCELoss().cuda()

        valid_queue = torch.utils.data.DataLoader(
                self.valid_data, batch_size=1,
                pin_memory=False, num_workers=0)

        start = time.time()

        voicebank_map = None
        if self.dataset_name == 'Voicebank28':
            voicebank_map = load_voicebank_map('../../datasets/Voicebank28/log_testset.txt')

        test_metrics = test_step(
            valid_queue, self.model, criterion, self.model_type, self.window,
            dataset_name=self.dataset_name,
            voicebank_map=voicebank_map
        )

        print(
            f"Model:{self.model_type} | "
            f"train:{self.dataset_name} | "
            f"test:{self.test_data} | "
            f"test_auc:{test_metrics['auc']:.4f} | "
            f"test_acc:{test_metrics['acc']:.4f} | "
            f"test_precision:{test_metrics['precision']:.4f} | "
            f"test_recall:{test_metrics['recall']:.4f} | "
            f"test_f1:{test_metrics['f1']:.4f} | "
            f"test_f2:{test_metrics['f2']:.4f}"
        )

        return test_metrics

"""
    def train(self):
        criterion = nn.BCELoss().cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs, eta_min=1e-6)
        train_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size,
            pin_memory=False, num_workers=0, shuffle=True)

        valid_queue = torch.utils.data.DataLoader(
            self.valid_data, batch_size=self.batch_size,
            pin_memory=False, num_workers=0)

        early = 0
        best_single_valid = np.inf # the lower the better

        print(f"Starting training for {self.epochs} epochs")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Valid samples: {len(self.valid_data)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model: {self.model_type}")
        print(f"Dataset: {self.dataset_name}")

        for e in range(self.epochs):
            print(f"\nEpoch {e+1}/{self.epochs}")
            self.model.drop_path_prob = 0
            start = time.time()

            train_auc, train_obj = train_step(
                train_queue, self.model, criterion, optimizer, self.model_type)

            if e == 0:
                print(f'# params: {sum(p.numel() for p in self.model.parameters())}')
            
            valid_auc, valid_acc, valid_precision, valid_recall, valid_f1, valid_obj = valid_step(
                valid_queue, self.model, criterion, self.dataset_name, self.model_type)

            scheduler.step()

            print(
                f"Epoch {e+1}/{self.epochs} | "
                f"train_auc={train_auc:.4f} | train_loss={train_obj:.6f} | "
                f"valid_auc={valid_auc:.4f} | valid_acc={valid_acc:.4f} | "
                f"valid_precision={valid_precision:.4f} | valid_recall={valid_recall:.4f} | "
                f"valid_f1={valid_f1:.4f} | valid_loss={valid_obj:.6f}"
            )

            if best_single_valid > valid_obj:
                best_single_valid = valid_obj
                torch.save(self.model.state_dict(),
                           f'{self.save_path}/{e:03d}_{self.model_type}_{self.dataset_name}.pth')
                print(f"Saved new best model at epoch {e+1} with valid_loss={valid_obj:.6f}")
                early = 0
            else:
                early += 1

            if early == 10:
                print(f'epoch:{e+1}, best valid:{best_single_valid}')
                break
"""

def train_step(train_queue, model, criterion, optimizer, model_type, grad_clip):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.train()
    device = 'cuda'

    for step, (inputs, target) in tqdm.tqdm(enumerate(train_queue), total=len(train_queue)):
        inputs = inputs.to(device)
        target = target.to(device)
        target = target.type(torch.float32)
        optimizer.zero_grad()
        
        if model_type == 'STA':
            logits, pipe, attn = model(inputs)
            loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
        else:
            logits = model(inputs)
            loss = criterion(logits, target)
 
        preds.append(logits.view(-1).detach()) 
        targets.append(target.view(-1).detach())
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        n = inputs.size(0)
        objs.update(loss.item(), n)

        if (step + 1) % 100 == 0 or (step + 1) == len(train_queue):
          print(
              f"[train] batch {step+1}/{len(train_queue)} | "
              f"loss={loss.item():.6f} | "
              f"inputs={tuple(inputs.shape)} | target={tuple(target.shape)}"
          )

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    auc = roc_auc_score(targets, preds)
    del preds, targets
    return auc, objs.avg


def valid_step(valid_queue, model, criterion, dataset, model_type):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.eval()

    batch_size = 512
    device = 'cuda'
    if dataset == 'TIMIT':
        for i in range(4):
            for step, (inputs, target) in enumerate(valid_queue):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    target = target.to(device)
                    target = target.type(torch.float32)
                    if model_type != 'STA':
                        logits = model(inputs)
                        loss = criterion(logits, target)
                    elif model_type == 'STA':
                        logits, pipe, attn = model(inputs)
                        loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
                    n = inputs.size(0)
                    objs.update(loss.item(), n)

                    if (step + 1) % 100 == 0 or (step + 1) == len(valid_queue):
                      print(
                          f"[valid] batch {step+1}/{len(valid_queue)} | "
                          f"loss={loss.item():.6f} | "
                          f"inputs={tuple(inputs.shape)} | target={tuple(target.shape)}"
                      )

                    preds.append(logits.view(-1).detach())
                    targets.append(target.view(-1).detach())

    for step, (inputs, target) in enumerate(valid_queue):
        with torch.no_grad():
            inputs = inputs.to(device)
            target = target.to(device)
            target = target.type(torch.float32)
            if model_type != 'STA':
                logits = model(inputs)
                loss = criterion(logits, target)
            
            elif model_type == 'STA':
                logits, pipe, attn = model(inputs)
                loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
            
            n = inputs.size(0)
            objs.update(loss.item(), n)
            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())

            if (step + 1) % 100 == 0 or (step + 1) == len(valid_queue):
                    print(
                        f"[valid] batch {step+1}/{len(valid_queue)} | "
                        f"loss={loss.item():.6f} | "
                        f"inputs={tuple(inputs.shape)} | target={tuple(target.shape)}"
                    )

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    auc = roc_auc_score(targets, preds)

    pred_labels = (preds >= 0.5).int()
    true_labels = torch.round(targets).int()

    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    f2 = fbeta_score(true_labels, pred_labels, beta=2, zero_division=0)

    del preds, targets

    return auc, acc, precision, recall, f1, f2, objs.avg

"""
def test_step(valid_queue, model, criterion, model_type, window):
    preds, targets = [], []
    model.eval()
    batch_size = 512
    device = 'cuda'

    from tqdm import tqdm

    for step, (inputs, target) in tqdm(enumerate(valid_queue), total=len(valid_queue)):
        with torch.no_grad():
            inputs = inputs.to(device)
            target = target.to(device)
            target = target.type(torch.float32)

            logits = bdnn_ensemble_prediction(model, inputs, window, batch_size, model_type)

            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())

            if (step + 1) % 100 == 0 or (step + 1) == len(valid_queue):
                print(
                    f"[test] batch {step+1}/{len(valid_queue)} | "
                    f"inputs={tuple(inputs.shape)} | target={tuple(target.shape)}"
                )

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    true_labels = torch.round(targets).int()
    pred_labels = (preds >= 0.5).int()

    auc = roc_auc_score(true_labels, preds)
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    del preds, targets
    return auc, acc, precision, recall, f1
"""

def test_step(valid_queue, model, criterion, model_type, window, dataset_name=None, voicebank_map=None):
    preds, targets = [], []
    model.eval()
    batch_size = 512
    device = 'cuda'

    from tqdm import tqdm

    # VOiCES grouped buckets
    room_buckets = {}
    room_noise_buckets = {}

    # Voicebank grouped buckets
    noise_buckets = {}
    noise_snr_buckets = {}

    # ONDRI grouped buckets
    #ondri_cohort_buckets = {}
    #ondri_cohort_timepoint_buckets = {}
    #ondri_cohort_timepoint_activity_buckets = {}

    # ONDRI grouped buckets
    ondri_cohort_buckets = {}
    ondri_activity_buckets = {}
    ondri_cohort_activity_buckets = {}
    ondri_task_type_buckets = {}

    for step, batch in tqdm(enumerate(valid_queue), total=len(valid_queue)):
        with torch.no_grad():
            inputs, target, meta = batch

            inputs = inputs.to(device)
            target = target.to(device)
            target = target.type(torch.float32)

            logits = bdnn_ensemble_prediction(model, inputs, window, batch_size, model_type)

            pred_np = logits.view(-1).detach().cpu().numpy()
            target_np = target.view(-1).detach().cpu().numpy()

            preds.append(pred_np)
            targets.append(target_np)

            # DataLoader batches dict values into lists
            audio_path = meta["audio_path"][0]

            if dataset_name == "VOiCES":
                room, noise = parse_voices_room_and_noise(audio_path)

                if room not in room_buckets:
                    room_buckets[room] = {"preds": [], "targets": []}
                room_buckets[room]["preds"].append(pred_np)
                room_buckets[room]["targets"].append(target_np)

                room_noise_key = (room, noise)
                if room_noise_key not in room_noise_buckets:
                    room_noise_buckets[room_noise_key] = {"preds": [], "targets": []}
                room_noise_buckets[room_noise_key]["preds"].append(pred_np)
                room_noise_buckets[room_noise_key]["targets"].append(target_np)

            elif dataset_name == "Voicebank28" and voicebank_map is not None:
                utt_id = parse_voicebank_utt_id(audio_path)
                meta_info = voicebank_map.get(utt_id, {"noise": "unknown", "snr": None})
                noise = meta_info["noise"]
                snr = meta_info["snr"]

                if noise not in noise_buckets:
                    noise_buckets[noise] = {"preds": [], "targets": []}
                noise_buckets[noise]["preds"].append(pred_np)
                noise_buckets[noise]["targets"].append(target_np)

                snr_key = format_snr_key(snr) if snr is not None else "snr_unknown"
                noise_snr_key = (noise, snr_key)
                
                if noise_snr_key not in noise_snr_buckets:
                    noise_snr_buckets[noise_snr_key] = {"preds": [], "targets": []}
                noise_snr_buckets[noise_snr_key]["preds"].append(pred_np)
                noise_snr_buckets[noise_snr_key]["targets"].append(target_np)

            elif dataset_name in {"ONDRI-DDK", "ONDRI-NARRATIVE"}:
                ondri = parse_ondri_metadata(audio_path)

                cohort = ondri["cohort"]
                activity = ondri["activity"]
                task_type = ondri["task_type"]

                # Per cohort
                add_to_metric_bucket(
                    ondri_cohort_buckets,
                    cohort,
                    pred_np,
                    target_np,
                )

                # Per activity
                add_to_metric_bucket(
                    ondri_activity_buckets,
                    activity,
                    pred_np,
                    target_np,
                )

                # Per cohort + activity
                add_to_metric_bucket(
                    ondri_cohort_activity_buckets,
                    (cohort, activity),
                    pred_np,
                    target_np,
                )

                # Combined AMR and SMR
                add_to_metric_bucket(
                    ondri_task_type_buckets,
                    task_type,
                    pred_np,
                    target_np,
                )
                
            extra = ""
            if dataset_name == "VOiCES":
                extra = f" room={room} | noise={noise}"
            elif dataset_name == "Voicebank28" and voicebank_map is not None:
                extra = f" noise={noise} | snr={snr}"

            elif dataset_name in {"ONDRI-DDK", "ONDRI-NARRATIVE"}:
                extra = (
                    f" cohort={cohort} | "
                    f"activity={activity} | "
                    f"task_type={task_type}"
                )
            
            """
            elif dataset_name == "ONDRI-DDK":
                extra = f" cohort={cohort} | timepoint={timepoint} | activity={activity}"
            """

            if (step + 1) % 100 == 0 or (step + 1) == len(valid_queue):
                print(
                        f"[test] batch {step+1}/{len(valid_queue)} | "
                        f"inputs={tuple(inputs.shape)} | target={tuple(target.shape)}"
                        f"{extra}"
                )

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    global_metrics = compute_binary_metrics(preds, targets)

    print("\nGlobal Test Metrics")
    print(
        f"auc={global_metrics['auc']:.4f} | "
        f"acc={global_metrics['acc']:.4f} | "
        f"precision={global_metrics['precision']:.4f} | "
        f"recall={global_metrics['recall']:.4f} | "
        f"f1={global_metrics['f1']:.4f} | "
        f"f2={global_metrics['f2']:.4f} | "
        f"miss_rate={global_metrics['miss_rate']:.4f} | "
        f"false_alarm_rate={global_metrics['false_alarm_rate']:.4f}"
    )
    print(
        f"TP={global_metrics['tp']} | TN={global_metrics['tn']} | "
        f"FP={global_metrics['fp']} | FN={global_metrics['fn']}"
    )

    if dataset_name == "Voicebank28":
        print("\nNoise type + SNR type")
        for noise, snr_key in sorted(noise_snr_buckets.keys()):
            combo_preds = np.concatenate(noise_snr_buckets[(noise, snr_key)]["preds"], axis=0)
            combo_targets = np.concatenate(noise_snr_buckets[(noise, snr_key)]["targets"], axis=0)
            m = compute_binary_metrics(combo_preds, combo_targets)
            print(
                f"{noise} | {snr_key} | "
                f"auc={m['auc']:.4f} | acc={m['acc']:.4f} | "
                f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | f1={m['f1']:.4f}"
            )

    elif dataset_name == 'VOiCES':
        print("\n Room type + noise type")
        for room, noise in sorted(room_noise_buckets.keys()):
            combo_preds = np.concatenate(room_noise_buckets[(room, noise)]["preds"], axis=0)
            combo_targets = np.concatenate(room_noise_buckets[(room, noise)]["targets"], axis=0)
            m = compute_binary_metrics(combo_preds, combo_targets)
            print(
                f"{room} | {noise} | "
                f"auc={m['auc']:.4f} | acc={m['acc']:.4f} | "
                f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | f1={m['f1']:.4f}"
            )

    elif dataset_name == "ONDRI-DDK":
        cohort_order = ["ALS", "FTD", "PD", "VCI"]
        activity_order = [
            "puh",
            "tuh",
            "kuh",
            "puhtuhkuh",
            "buttercup",
            "unknown",
        ]
        task_type_order = [
            "AMR",
            "SMR",
            "unknown",
        ]

        print("\nPer cohort")
        for cohort in cohort_order:
            if cohort not in ondri_cohort_buckets:
                continue

            print_metric_bucket(
                cohort,
                ondri_cohort_buckets[cohort],
            )

        print("\nPer activity")
        for activity in activity_order:
            if activity not in ondri_activity_buckets:
                continue

            print_metric_bucket(
                activity,
                ondri_activity_buckets[activity],
            )

        print("\nCombined AMR and SMR")
        for task_type in task_type_order:
            if task_type not in ondri_task_type_buckets:
                continue

            print_metric_bucket(
                task_type,
                ondri_task_type_buckets[task_type],
            )

        print("\nPer cohort + activity")
        for cohort in cohort_order:
            for activity in activity_order:
                key = (cohort, activity)

                if key not in ondri_cohort_activity_buckets:
                    continue

                print_metric_bucket(
                    f"{cohort} | {activity}",
                    ondri_cohort_activity_buckets[key],
                )

    elif dataset_name == "ONDRI-NARRATIVE":
        cohort_order = [
            "ADMCI",
            "ALS",
            "FTD",
            "PD",
            "VCI",
        ]
    
        activity_order = [
            "seqstort1",
            "seqstort2a",
        ]
    
        print("\nPer cohort")
        for cohort in cohort_order:
            if cohort not in ondri_cohort_buckets:
                continue
    
            print_metric_bucket(
                cohort,
                ondri_cohort_buckets[cohort],
            )
    
        print("\nPer activity")
        for activity in activity_order:
            if activity not in ondri_activity_buckets:
                continue
    
            print_metric_bucket(
                activity,
                ondri_activity_buckets[activity],
            )
    
        print("\nPer cohort + activity")
        for cohort in cohort_order:
            for activity in activity_order:
                key = (cohort, activity)
    
                if key not in ondri_cohort_activity_buckets:
                    continue
    
                print_metric_bucket(
                    f"{cohort} | {activity}",
                    ondri_cohort_activity_buckets[key],
                )
    
        """
    elif dataset_name == "ONDRI-DDK":
        print("\nPer cohort + timepoint + activity")
        for cohort, timepoint, activity in sorted(ondri_cohort_timepoint_activity_buckets.keys()):
            combo_preds = np.concatenate(
                ondri_cohort_timepoint_activity_buckets[(cohort, timepoint, activity)]["preds"], axis=0
            )
            combo_targets = np.concatenate(
                ondri_cohort_timepoint_activity_buckets[(cohort, timepoint, activity)]["targets"], axis=0
            )
            m = compute_binary_metrics(combo_preds, combo_targets)
            print(
                f"{cohort} | {timepoint} | {activity} | "
                f"auc={m['auc']:.4f} | acc={m['acc']:.4f} | "
                f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | f1={m['f1']:.4f}"
            )
            
        print("\nPer cohort + timepoint")
        for (cohort, timepoint) in sorted(ondri_cohort_timepoint_buckets.keys()):
            combo_preds = np.concatenate(
                ondri_cohort_timepoint_buckets[(cohort, timepoint)]["preds"], axis=0
            )
            combo_targets = np.concatenate(
                ondri_cohort_timepoint_buckets[(cohort, timepoint)]["targets"], axis=0
            )
            m = compute_binary_metrics(combo_preds, combo_targets)

            print(
                f"{cohort} | {timepoint} | "
                f"auc={m['auc']:.4f} | acc={m['acc']:.4f} | "
                f"precision={m['precision']:.4f} | recall={m['recall']:.4f} | "
                f"f1={m['f1']:.4f} | miss_rate={m['miss_rate']:.4f} | "
                f"false_alarm_rate={m['false_alarm_rate']:.4f}"
            )

    return (
        global_metrics["auc"],
        global_metrics["acc"],
        global_metrics["precision"],
        global_metrics["recall"],
        global_metrics["f1"],
    )
    """

    return global_metrics


class VAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, label_files, n_fft=400, n_mels=80, sample_rate=16000, mode='train',
                 snr_low=-10, snr_high=10, train_portion=1, window=[-19, -9, -1, 0, 1, 9, 19], model_type='NewSearch'):

        self.audio_paths = list(audio_files)
        self.label_paths = list(label_files)
        self.audio_files = audio_files
        self.label_files = label_files
        self.mode = mode

        loaded_audio = []
        loaded_labels = []

        print(f"[{self.mode}] Loading {len(self.audio_files)} spectrogram files and labels...")

        for i, (audio_path, label_path) in enumerate(zip(self.audio_files, self.label_files), 1):
            audio_arr = np.load(audio_path)
            label_arr = np.load(label_path)

            loaded_audio.append(torch.from_numpy(audio_arr))
            loaded_labels.append(torch.from_numpy(label_arr))

            if i % 100 == 0 or i == len(self.audio_files):
                print(
                    f"[{self.mode}] Loaded {i}/{len(self.audio_files)} | "
                    f"spec={os.path.basename(audio_path)} shape={audio_arr.shape} | "
                    f"label={os.path.basename(label_path)} shape={label_arr.shape}"
                )

        self.audio_files = loaded_audio
        self.label_files = loaded_labels

        self.n_voices = len(self.audio_files)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.window = torch.tensor(window)
        self.window -= self.window.min()
        self.sample_rate = sample_rate
        self.melscale = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        ).cuda()
        self.n_frame = 63
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.model_type = model_type
        self.mask = torchaudio.transforms.FrequencyMasking(int(0.3 * n_mels)).cuda()
         
    def __len__(self):
        return self.n_voices

    def __getitem__(self, idx):
        v_name, l_name = self.audio_files[idx], self.label_files[idx]
        #voice = torch.from_numpy(np.load(v_name)).cuda()
        voice = v_name.cuda()
        # if self.mode == 'train':
        #     weight = torch.pow(10., torch.rand([])*1/2 - 1/4) # [-1/4, 1/4]
        #     voice *= weight.to(voice.dtype)
        voice = torch.squeeze(voice)
        # label = torch.from_numpy(np.load(l_name)).cuda()
        label = l_name.cuda()
        label = label[:voice.size(1)]

        if idx < 3:
          print(
              f"[{self.mode}] Sample check idx={idx} | "
              f"voice_shape={tuple(voice.shape)} | "
              f"label_shape={tuple(label.shape)}"
          )

        assert label.shape[0] == voice.shape[1]
        if self.mode != 'test':
            voice, label = self.slice(voice, label)
        voice = torch.transpose(voice, 0, 1) # T * C
        voice = voice.type(torch.float32)
        audio = torch.matmul(torch.abs(voice), self.melscale)
        audio = torch.log10(torch.clamp(audio, min=1e-10))

        if self.mode == 'train':
            # up and down
            audio += (torch.rand([])*1/2 - 1/4).to(audio.dtype) # [-1/4, 1/4]
            # freq masking
            audio = torch.transpose(audio, 0, 1)
            audio = torch.unsqueeze(audio, 0)
            audio = self.mask(audio)
            audio = torch.squeeze(audio)
            audio = torch.transpose(audio, 0, 1)

        audio = torch.unsqueeze(audio, 0)
        audio = audio.to(torch.float32) # batch, winodw(time), freq
        
        if self.mode == 'test':
          meta = {
                "audio_path": self.audio_paths[idx],
                "label_path": self.label_paths[idx],
          }
          return audio, label, meta

        return audio, label

    def slice(self, spec, label=None):
        # [chan, time, freq]
        time = spec.size(1)
        offset = torch.randint(high=max([1, time-self.n_frame]), size=[]).cuda()
        window = self.window.cuda() + offset
        spec = torch.index_select(spec, 1, window)
        if label is not None:
            label = torch.index_select(label, 0, window)
            return spec, label
        return spec

    def synthesize(self, voice, noise):
        # SNR
        weight = torch.pow(
            10., (torch.rand([])*(self.snr_high-self.snr_low)+self.snr_low)/20)
        weight = weight.to(voice.dtype)
        audio = (noise + voice * weight) / (1 + weight)
 
        # dB
        weight = torch.pow(10., torch.rand([])*1/2 - 1/4) # [-1/4, 1/4]
        audio *= weight.to(audio.dtype)

        return audio


def bdnn_ensemble_prediction(model, spectrogram, window, batch_size, model_type):
    spectrogram = torch.squeeze(spectrogram, 0)
    assert spectrogram.dim() == 3 # [chan, time, freq]
    model.eval()

    # sequence to slices
    window = torch.tensor(window)
    window -= window.min()
    win_width = window.max()
    slices = []
    for w in window:
        if w == win_width:
            slices.append(spectrogram[:, win_width:]) # [chan, time-win_width, freq]
        else:
            slices.append(spectrogram[:, w:-win_width+w])
    slices = torch.stack(slices, axis=0) # [win_size, chan, time-win_width, freq]
    slices = torch.transpose(slices, 0, 2)

    # inference
    predictions = []
    for i in range(int(np.ceil(slices.size(0) / batch_size))):
        inputs = slices[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            inputs = inputs.cuda()
            if model_type != 'STA':
                prediction = model(inputs)
            elif model_type == 'STA':
                prediction, _, _ = model(inputs)
            if len(prediction.shape) != 2:
                prediction = torch.unsqueeze(prediction, 0)
            predictions.append(prediction) # appending only preds
    del slices
    try:
        predictions = torch.cat(predictions, dim=0)
    except:
        import pdb; pdb.set_trace()

    # slices to sequence
    n_frames = spectrogram.size(1) 
    outputs = torch.zeros([n_frames], dtype=torch.float32).cuda()
    total_counts = torch.zeros([n_frames], dtype=torch.float32).cuda()
    
    for i, w in enumerate(window):
        if w == win_width:
            outputs[win_width:] += predictions[:, i]
            total_counts[win_width:] += 1
        else:
            outputs[w:-win_width+w] += predictions[:, i]
            total_counts[w:-win_width+w] += 1
    return outputs / (total_counts + 1e-8)


def get_model(model_type, dataset_name, mode, n_mels, save_path, init_checkpoint=None,sa_dropout=0.0):
    if model_type == 'BDNN':
        model = bDNN().cuda()
    elif model_type == 'ACAM':
        model = ACAM(n_mels).cuda()
    elif model_type == 'STA':
        model = LeeVAD(n_mels).cuda()
    elif model_type == 'SL_model':
        model = SelfAttentiveVAD(n_mels, dropout=sa_dropout).cuda()

    elif model_type =='Darts2D':
        genotype = Genotype(normal=[('zero_original', 0), ('skip_connect_original', 1),
                                    ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                    ('skip_connect_original', 1), ('avg_pool_3x3', 0),
                                    ('zero_original', 2), ('sep_conv_3x3_original', 4)],
                            normal_concat=range(2, 6),
                            reduce=[('zero_original', 0), ('skip_connect_original', 1),
                                    ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                    ('skip_connect_original', 1), ('avg_pool_3x3', 0),
                                    ('zero_original', 2), ('sep_conv_3x3_original', 4)],
                            reduce_concat=range(2, 6))
        model = NetworkVADOriginal(16, 8, genotype, use_second=False).cuda()
    elif model_type == 'NewSearch':
        genotype = Genotype(normal=[('MHA2D_F_4', 0), ('MBConv_5x5_x4', 1),
                                    ('MBConv_5x5_x4', 2), ('MHA2D_F_2', 0),
                                    ('SE_0.25', 2), ('MBConv_3x3_x4', 0)],
                            normal_concat=range(2, 5),
                            reduce=[('MHA2D_F_4', 0), ('MBConv_5x5_x4', 1),
                                    ('MBConv_5x5_x4', 2), ('MHA2D_F_2', 0),
                                    ('SE_0.25', 2), ('MBConv_3x3_x4', 0)],
                            reduce_concat=range(2, 5))
        model = NetworkVADv2(28, 4, genotype, False, 0, False, 7, n_mels).cuda()    

    else:
        raise ValueError(
            f'Unsupported model type: {model_type}'
        )

    checkpoint_path = None

    # New feature. Allows us to fine-tune by getting pre-trained weights by loading a specific checkpoint.
    if init_checkpoint is not None:
        checkpoint_path = init_checkpoint

    # Preserves the existing testing behaviour; when no specific checkpoint is provided, uses the latest checkpoint found inside save_path.
    elif mode == 'test':
        checkpoint_candidates = sorted(glob(os.path.join(save_path, '*.pth')))

        if not checkpoint_candidates:
            raise FileNotFoundError(
                f'No .pth checkpoints found in save_path: {save_path}'
            )

        checkpoint_path = checkpoint_candidates[-1]

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

        state_dict = torch.load(
            checkpoint_path,
            map_location='cuda',
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        print(f'{checkpoint_path}')
        print(
            f'Trainable parameters: '
            f'{sum(p.numel() for p in model.parameters() if p.requires_grad)}'
        )

    else:
        print(
            f'No initialization checkpoint supplied. '
            f'{model_type} will use newly initialized weights.'
        )

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NewSearch')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='TIMIT',
                        choices=['ONDRI-SILERO', 'ONDRI-NARRATIVE', 'TIMIT', 'CV', 'VOiCES', 'MS-SNSD', 'Voicebank28', 'ONDRI-DDK', 'nasvad-subset', 'nasvad-subset10', 'nasvad-subset50', 'TRAIN', 'ONDRI-DDK-Old'])
    parser.add_argument('--test_dataset', type=str, default='TIMIT')
    parser.add_argument('--test_path', type=str, default='/nfs/roberts/Humzah-Workspace/USRI_test/TEST', help='Path to the ONDRI test dataset',
    )
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--init_checkpoint',type=str,default=None,help=('Optional .pth checkpoint used to initialize the model and allow for model fine-tuning during training. '),)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3,)
    parser.add_argument("--weight_decay", type=float, default=0.0,)
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce","weighted_bce","focal"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--sa_dropout", type=float, default=0.0)
 
    args = parser.parse_args()

    model = get_model(args.model, args.dataset, args.mode, args.n_mels, args.save_path, init_checkpoint=args.init_checkpoint, sa_dropout=args.sa_dropout)
    
    trainer_args = {'dataset': args.dataset,
                    'test_dataset': args.test_dataset,
                    'window': [-19, -9, -1, 0, 1, 9, 19],
                    'mode': args.mode,
                    'model_type': args.model, 'model': model,
                    'n_mels': args.n_mels,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'loss_type': args.loss_type,
                    'patience': args.patience,
                    'grad_clip': args.grad_clip,
                    'drop_path': args.drop_path,
                    }

    datapath_mapper = {
        # Add mapping for ONDRI fine-tuning dataset (About 512 participants; Confirm if DDK files only or narrative files as well; Need to create using Silero)
        'train': {
            'ONDRI-SILERO': '../../datasets/ondri_silero_train,../../datasets/ondri_silero_valid',
            'TIMIT': 'datasets/make_nasvad/train,datasets/make_nasvad/valid',
            'CV':    'datasets/make_nasvad/train,datasets/make_nasvad/valid',
            'VOiCES': '../../datasets/VOICES_two/TRAIN,../../datasets/VOICES_two/VALID',
            'MS-SNSD': '../../datasets/MS-SNSD_two/TRAIN,../../datasets/MS-SNSD_two/VALID',
            'Voicebank28': '../../datasets/nasvad_subsets/train_1pct.jsonl,../../datasets/nasvad_subsets/valid_full.jsonl',
            'nasvad-subset': '../../datasets/nasvad_subsets/train_1pct.jsonl,../../datasets/nasvad_subsets/valid_full.jsonl',
            'nasvad-subset10': '../../datasets/nasvad_subsets/train_10pct.jsonl,../../datasets/nasvad_subsets/valid_full.jsonl',
            'nasvad-subset50': '../../datasets/nasvad_subsets/train_50pct.jsonl,../../datasets/nasvad_subsets/valid_full.jsonl',
            'TRAIN': '/content/datasets/datasets/TRAIN,/content/datasets/datasets/VALID',
        },
        'test': {
            'TIMIT': 'datasets/make_nasvad/train,datasets/make_nasvad/test',
            'CV':    'datasets/make_nasvad/train,datasets/make_nasvad/test',
            'VOiCES': '../../datasets/VOICES_two/TRAIN,../../datasets/VOiCES/test/TEST',
            'Voicebank28': '../../datasets/Voicebank28_two/TRAIN,../../datasets/Voicebank28/test/TEST',
            'ONDRI-DDK': f'{args.test_path},{args.test_path}',
            'ONDRI-DDK-Old': '../../datasets/ONDRI_DDK_Data,../../datasets/ONDRI_DDK_Test',
            'ONDRI-NARRATIVE': f'{args.test_path},{args.test_path}',
        }
    }

    if args.mode =='train':
        trainer = Trainer(datapath_mapper[args.mode][args.dataset],
                          args.save_path,
                          epochs=args.epochs,
                          **trainer_args)
        trainer.train()
    else:
        """
        for dataset in sorted(datapath_mapper['test'].keys()):
            print(dataset)
            trainer = Trainer(datapath_mapper[args.mode][dataset],
                              args.save_path, epochs=None, **trainer_args)
            trainer.test()
        """

        print(args.dataset)
        trainer = Trainer(datapath_mapper[args.mode][args.dataset],
                            args.save_path, epochs=None, **trainer_args)
        trainer.test()
