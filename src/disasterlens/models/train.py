
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# Configuration
# ============================================================

SEED = 42
IMAGE_SIZE = 128
FEATURE_BATCH_SIZE = 8
HEAD_BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 8
NUM_WORKERS = 0

ROOT = Path("data/raw/triage")
ARTIFACTS = Path("artifacts")

# New checkpoint names: existing model files remain untouched.
BEST_MODEL_PATH = ARTIFACTS / "triage_image_modality_v1.pt"
METRICS_PATH = ARTIFACTS / "triage_image_modality_v1_metrics.json"

ACTIONS = ["discard", "defer", "downlink_now"]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}

# Explicit, stable encoding. Unknown values are rejected.
MODALITIES = ["optical-to-SAR", "optical-to-optical"]
MODALITY_TO_ID = {name: i for i, name in enumerate(MODALITIES)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_image_path(image_path):
    path = Path(image_path)
    return path if path.is_absolute() else ROOT / path


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def load_rgb(self, relative_path):
        path = resolve_image_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        row = self.rows[index]

        baseline = self.transform(self.load_rgb(row["baseline_image"]))
        current = self.transform(self.load_rgb(row["current_image"]))
        images = torch.cat([baseline, current], dim=0)

        action = row["target_output"]["action"]
        modality = row.get("modality")

        if action not in ACTION_TO_ID:
            raise ValueError(f"Unknown action: {action}")
        if modality not in MODALITY_TO_ID:
            raise ValueError(f"Unknown or missing modality: {modality}")

        return (
            images,
            torch.tensor(ACTION_TO_ID[action], dtype=torch.long),
            torch.tensor(MODALITY_TO_ID[modality], dtype=torch.long),
        )


class TriageModel(nn.Module):
    def __init__(self, num_classes=3, num_modalities=2):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Three image feature groups (baseline, current, difference),
        # plus a two-element one-hot modality vector.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 + num_modalities, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def extract_image_features(self, images):
        with torch.no_grad():
            baseline = images[:, :3]
            current = images[:, 3:]

            baseline_features = self.pool(
                self.encoder(baseline)
            ).flatten(1)

            current_features = self.pool(
                self.encoder(current)
            ).flatten(1)

            difference = torch.abs(
                current_features - baseline_features
            )

            return torch.cat(
                [baseline_features, current_features, difference],
                dim=1,
            )


def cache_features(model, loader, device, name):
    model.eval()
    image_features, labels, modality_ids = [], [], []

    with torch.inference_mode():
        for images, targets, modalities in loader:
            images = images.to(device)
            features = model.extract_image_features(images)

            image_features.append(features.cpu())
            labels.append(targets.cpu())
            modality_ids.append(modalities.cpu())

    image_features = torch.cat(image_features)
    labels = torch.cat(labels)
    modality_ids = torch.cat(modality_ids)

    print(f"{name} image features: {tuple(image_features.shape)}")
    return image_features, labels, modality_ids


def combine_features(image_features, modality_ids):
    # One-hot modality encoding, concatenated with cached image features.
    modality_one_hot = torch.nn.functional.one_hot(
        modality_ids, num_classes=len(MODALITIES)
    ).float()

    return torch.cat([image_features, modality_one_hot], dim=1)


def evaluate(model, features, labels, device):
    model.eval()

    with torch.inference_mode():
        logits = model.classifier(features.to(device))
        predictions = logits.argmax(dim=1).cpu().tolist()

    targets = labels.tolist()

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets,
            predictions,
            labels=list(range(len(ACTIONS))),
            average="macro",
            zero_division=0,
        )),
        "targets": targets,
        "predictions": predictions,
    }


def print_results(title, results):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        results["targets"],
        results["predictions"],
        labels=list(range(len(ACTIONS))),
        target_names=ACTIONS,
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        results["targets"],
        results["predictions"],
        labels=list(range(len(ACTIONS))),
    ))


def main():
    set_seed()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_path = ROOT / "train_flat.jsonl"
    eval_path = ROOT / "eval_flat.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing dataset: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing dataset: {eval_path}")

    all_train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    if not all_train_rows or not eval_rows:
        raise ValueError("Training or evaluation data is empty.")

    for row in all_train_rows + eval_rows:
        if row.get("modality") not in MODALITY_TO_ID:
            raise ValueError(
                f"Unexpected modality in example {row.get('example_id')}: "
                f"{row.get('modality')}"
            )

    all_labels = [row["target_output"]["action"] for row in all_train_rows]

    train_indices, val_indices = train_test_split(
        np.arange(len(all_train_rows)),
        test_size=0.20,
        random_state=SEED,
        stratify=all_labels,
    )

    train_rows = [all_train_rows[i] for i in train_indices]
    val_rows = [all_train_rows[i] for i in val_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")
    print(f"Event-held-out evaluation examples: {len(eval_rows)}")
    print("Training labels:", Counter(
        r["target_output"]["action"] for r in train_rows
    ))
    print("Training modalities:", Counter(r["modality"] for r in train_rows))
    print("Evaluation modalities:", Counter(r["modality"] for r in eval_rows))

    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        PairDataset(val_rows),
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        PairDataset(eval_rows),
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = TriageModel(
        num_classes=len(ACTIONS),
        num_modalities=len(MODALITIES),
    ).to(device)
    model.eval()

    print("\nExtracting frozen image features...")
    train_img, train_y, train_mod = cache_features(
        model, train_loader, device, "Training"
    )
    val_img, val_y, val_mod = cache_features(
        model, val_loader, device, "Validation"
    )
    eval_img, eval_y, eval_mod = cache_features(
        model, eval_loader, device, "Evaluation"
    )

    train_x = combine_features(train_img, train_mod)
    val_x = combine_features(val_img, val_mod)
    eval_x = combine_features(eval_img, eval_mod)

    # Balance the training sampler only. No class weights are applied
    # simultaneously, avoiding double compensation for rare classes.
    train_label_ids = train_y.tolist()
    counts = Counter(train_label_ids)
    sample_weights = torch.tensor(
        [1.0 / counts[label] for label in train_label_ids],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_label_ids),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    head_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=HEAD_BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nTraining classifier with image + modality features...")
    for epoch in range(1, EPOCHS + 1):
        model.classifier.train()
        total_loss = 0.0
        total_examples = 0

        for features, labels in head_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_examples += batch_n

        val_results = evaluate(model, val_x, val_y, device)
        average_loss = total_loss / max(total_examples, 1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Val accuracy: {val_results['accuracy']:.4f} | "
            f"Val macro-F1: {val_results['macro_f1']:.4f}"
        )

        if val_results["macro_f1"] > best_val_f1:
            best_val_f1 = val_results["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Saved checkpoint: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best epoch was {best_epoch}."
            )
            break

    state = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")

    val_results = evaluate(model, val_x, val_y, device)
    print_results("VALIDATION RESULTS", val_results)

    # This evaluation set was inspected in previous experiments and is
    # not an untouched benchmark.
    eval_results = evaluate(model, eval_x, eval_y, device)
    print_results("EVENT-HELD-OUT EVALUATION RESULTS", eval_results)

    metrics = {
        "experiment": "frozen_resnet18_plus_modality",
        "seed": SEED,
        "modalities": MODALITIES,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "validation_accuracy": val_results["accuracy"],
        "validation_macro_f1": val_results["macro_f1"],
        "evaluation_accuracy": eval_results["accuracy"],
        "evaluation_macro_f1": eval_results["macro_f1"],
        "actions": ACTIONS,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "evaluation_examples": len(eval_rows),
        "checkpoint": str(BEST_MODEL_PATH),
        "note": (
            "Experimental auxiliary dataset. Evaluation split was inspected "
            "in prior experiments and is not untouched. Modality must be "
            "known at inference time."
        ),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCheckpoint saved: {BEST_MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
