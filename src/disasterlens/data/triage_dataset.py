
import json
from pathlib import Path

from PIL import Image


class TriageDataset:
    """Load paired satellite images and triage labels from JSONL."""

    VALID_ACTIONS = {"discard", "defer", "downlink_now"}

    def __init__(self, root_dir: str | Path, split: str = "train"):
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")

        self.root_dir = Path(root_dir)
        manifest = self.root_dir / f"{split}_flat.jsonl"

        if not manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest}")

        self.records = []

        with manifest.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)
                target = record.get("target_output", {})
                action = target.get("action")

                if action not in self.VALID_ACTIONS:
                    raise ValueError(
                        f"Invalid action on line {line_number}: {action}"
                    )

                for key in ("baseline_image", "current_image"):
                    image_path = self.root_dir / record[key]
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Image missing on line {line_number}: {image_path}"
                        )

                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        baseline_path = self.root_dir / record["baseline_image"]
        current_path = self.root_dir / record["current_image"]

        with Image.open(baseline_path) as image:
            baseline = image.convert("RGB")

        with Image.open(current_path) as image:
            current = image.convert("RGB")

        return {
            "example_id": record["example_id"],
            "baseline_image": baseline,
            "current_image": current,
            "action": record["target_output"]["action"],
            "category": record["target_output"]["category"],
            "rationale": record["target_output"]["rationale"],
            "bbox_norm": record["target_output"]["bbox_norm"],
            "modality": record["modality"],
        }
