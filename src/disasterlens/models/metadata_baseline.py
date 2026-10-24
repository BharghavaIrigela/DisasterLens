
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

ROOT = Path("data/raw/triage")
ACTIONS = ["discard", "defer", "downlink_now"]
SEED = 42


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def features_from_rows(rows):
    features = []

    for row in rows:
        modality = row["modality"]

        features.append([
            float(row["damage_ratio"]),
            float(row["destruction_ratio"]),
            1.0 if modality == "optical-to-SAR" else 0.0,
        ])

    return np.asarray(features, dtype=np.float32)


def main():
    train_path = ROOT / "train_flat.jsonl"
    eval_path = ROOT / "eval_flat.jsonl"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Expected train_flat.jsonl and eval_flat.jsonl in {ROOT}"
        )

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    X_train = features_from_rows(train_rows)
    X_eval = features_from_rows(eval_rows)

    y_train = [row["target_output"]["action"] for row in train_rows]
    y_eval = [row["target_output"]["action"] for row in eval_rows]

    print("Training examples:", len(train_rows))
    print("Evaluation examples:", len(eval_rows))
    print("Training label counts:", Counter(y_train))
    print("Evaluation label counts:", Counter(y_eval))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    print("\n=== METADATA BASELINE RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_eval, predictions):.4f}")
    print(
        "Macro-F1:",
        f"{f1_score(y_eval, predictions, labels=ACTIONS, average='macro', zero_division=0):.4f}",
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_eval,
            predictions,
            labels=ACTIONS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_eval, predictions, labels=ACTIONS))

    print("\nFeature importances:")
    for name, importance in zip(
        ["damage_ratio", "destruction_ratio", "modality_is_optical_to_SAR"],
        model.feature_importances_,
    ):
        print(f"{name}: {importance:.4f}")

    print(
        "\nCaution: this is a metadata diagnostic baseline, not an "
        "image-only model. Its scores are not directly comparable to "
        "the ResNet-18 model."
    )


if __name__ == "__main__":
    main()
