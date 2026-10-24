
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageChops
from torch import nn
from torchvision.models import resnet18
from torchvision import transforms


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "triage_gpu_v1.pt"

ACTIONS = ["discard", "defer", "downlink_now"]
IMAGE_SIZE = 128

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ------------------------------------------------------------
# Model
# Architecture expected by the existing GPU checkpoint.
# ------------------------------------------------------------

class TriageModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):
        super().__init__()

        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images):
        baseline = images[:, :3, :, :]
        current = images[:, 3:, :, :]

        baseline_features = self.pool(
            self.encoder(baseline)
        ).flatten(1)

        current_features = self.pool(
            self.encoder(current)
        ).flatten(1)

        difference = torch.abs(
            current_features - baseline_features
        )

        combined = torch.cat(
            [baseline_features, current_features, difference],
            dim=1,
        )

        return self.classifier(combined)


# ------------------------------------------------------------
# Load the trained model once
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint is not a state dictionary.")

    # Handle common checkpoint wrappers and DataParallel prefixes.
    for wrapper_key in ("state_dict", "model_state_dict"):
        if wrapper_key in checkpoint:
            checkpoint = checkpoint[wrapper_key]
            break

    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    first_layer = state.get("classifier.0.weight")
    last_layer = state.get("classifier.3.weight")

    if first_layer is None or last_layer is None:
        raise ValueError(
            "Checkpoint architecture does not match this application. "
            "Expected classifier.0.weight and classifier.3.weight."
        )

    if first_layer.shape[1] != 1536:
        raise ValueError(
            "This checkpoint does not match the expected "
            "1536-dimensional paired-image architecture."
        )

    if last_layer.shape[0] != len(ACTIONS):
        raise ValueError("Unexpected number of output classes.")

    model = TriageModel(
        hidden_dim=first_layer.shape[0],
        num_classes=last_layer.shape[0],
    )

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ------------------------------------------------------------
# Inference and report generation
# ------------------------------------------------------------

def predict(model, baseline_image, current_image):
    baseline_tensor = PREPROCESS(
        baseline_image.convert("RGB")
    )
    current_tensor = PREPROCESS(
        current_image.convert("RGB")
    )

    pair = torch.cat(
        [baseline_tensor, current_tensor], dim=0
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(pair)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax().item())

    scores = {
        action: round(float(probabilities[i]), 6)
        for i, action in enumerate(ACTIONS)
    }

    return {
        "action": ACTIONS[predicted_index],
        "confidence": scores[ACTIONS[predicted_index]],
        "probabilities": scores,
    }


def make_difference_image(baseline, current):
    baseline = baseline.convert("RGB")
    current = current.convert("RGB").resize(baseline.size)
    return ImageChops.difference(baseline, current)


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------

st.set_page_config(
    page_title="DisasterLens",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ DisasterLens")
st.subheader("AI-assisted satellite change assessment")

st.write(
    "Compare baseline and current satellite imagery to obtain "
    "an experimental disaster-triage prediction."
)

st.info(
    "Research prototype: predictions are not verified damage "
    "assessments. Always review imagery and corroborate results "
    "before making operational decisions."
)

with st.sidebar:
    st.header("System status")
    st.write(f"**Device:** {DEVICE}")
    st.write("**Model:** Paired-image ResNet-18")
    st.write("**Input:** Baseline and current RGB images")
    st.write("**Output:** Discard, defer, or downlink now")
    st.caption(
        "The model does not directly predict damage bounding boxes."
    )

    if torch.cuda.is_available():
        st.success("CUDA is available.")
    else:
        st.warning("Running inference on CPU.")

left, right = st.columns(2)

with left:
    st.markdown("### 1. Baseline image")
    baseline_file = st.file_uploader(
        "Upload the earlier satellite image",
        type=["png", "jpg", "jpeg", "webp"],
        key="baseline",
    )

with right:
    st.markdown("### 2. Current image")
    current_file = st.file_uploader(
        "Upload the later satellite image",
        type=["png", "jpg", "jpeg", "webp"],
        key="current",
    )

if baseline_file is not None and current_file is not None:
    try:
        baseline_image = Image.open(baseline_file).convert("RGB")
        current_image = Image.open(current_file).convert("RGB")

        st.markdown("### Image comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                baseline_image,
                caption=f"Baseline: {baseline_image.width} × "
                        f"{baseline_image.height}",
                use_container_width=True,
            )

        with col2:
            st.image(
                current_image,
                caption=f"Current: {current_image.width} × "
                        f"{current_image.height}",
                use_container_width=True,
            )

        with st.expander("View a simple pixel-difference visualization"):
            st.image(
                make_difference_image(baseline_image, current_image),
                caption=(
                    "Absolute pixel difference. This is a visual aid, "
                    "not a validated damage map."
                ),
                use_container_width=True,
            )

        st.caption(
            "The model resizes both images to 128 × 128. "
            "For meaningful comparisons, use corresponding geographic "
            "areas with compatible alignment and image characteristics."
        )

        if st.button(
            "Analyze image pair",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing the image pair..."):
                model = load_model()
                result = predict(
                    model,
                    baseline_image,
                    current_image,
                )

            st.session_state["disasterlens_result"] = result

    except Exception as exc:
        st.error(f"Could not process the images: {exc}")

result = st.session_state.get("disasterlens_result")

if result is not None:
    st.divider()
    st.markdown("## Assessment result")

    action = result["action"]

    labels = {
        "discard": "Discard",
        "defer": "Defer for further review",
        "downlink_now": "Downlink now",
    }

    st.markdown(f"### Predicted action: {labels[action]}")
    st.metric(
        "Model confidence",
        f"{result['confidence'] * 100:.2f}%",
    )

    st.caption(
        "Confidence is the model's softmax score, not a calibrated "
        "probability that the decision is correct."
    )

    st.markdown("### Class scores")

    for action_name, score in result["probabilities"].items():
        st.write(f"**{labels[action_name]}:** {score * 100:.2f}%")
        st.progress(score)

    if action == "defer":
        st.warning(
            "The model predicts defer. Obtain further review before "
            "making a triage decision."
        )
    elif action == "downlink_now":
        st.warning(
            "The model predicts downlink now. Review the imagery "
            "and independently verify the finding."
        )
    else:
        st.info(
            "The model predicts discard. Do not discard potentially "
            "important imagery solely on this prediction."
        )

    report = {
        "application": "DisasterLens",
        "model_checkpoint": "triage_gpu_v1.pt",
        "predicted_action": action,
        "confidence": result["confidence"],
        "class_scores": result["probabilities"],
        "device": str(DEVICE),
        "limitations": [
            "Experimental model trained on a small auxiliary dataset.",
            "Confidence scores are not calibrated probabilities.",
            "Predictions require human review.",
            "No validated damage localization is provided.",
        ],
    }

    st.download_button(
        "Download assessment report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="disasterlens_assessment.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "DisasterLens | AI-assisted disaster imagery triage | "
    "Experimental research prototype"
)
