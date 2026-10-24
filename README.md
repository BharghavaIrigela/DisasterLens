# DisasterLens

### Satellite-Based Disaster Damage Assessment & Response Prioritization

Compare pre-disaster and post-disaster imagery, inspect visual changes, and explore model-assisted response priorities.

<p align="center">
  <a href="https://disasterlensss.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_DisasterLens-2ea44f?style=for-the-badge" alt="Launch DisasterLens Live Demo">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/Status-Research_Prototype-blue" alt="Research Prototype">
</p>

**Live Demo:** [https://disasterlensss.streamlit.app/](https://disasterlensss.streamlit.app/)

**GitHub Repository:** [BharghavaIrigela/DisasterLens](https://github.com/BharghavaIrigela/DisasterLens)

---

## Overview

**DisasterLens** is an AI-powered research prototype designed to support satellite-image comparison and disaster-response prioritization. It analyzes imagery captured before and after a potential disaster, visualizes image-level changes, and presents model-generated triage scores to help users explore response priorities.

The application combines a Streamlit web interface with a trained PyTorch image-classification model. Users can upload a pair of images, inspect their differences, review the predicted action scores, and export a structured JSON report.

The goal is to explore how computer vision can support the interpretation of satellite imagery and help prioritize images for further analysis following disaster events.

## Live Demo

### Try DisasterLens

**[Launch the Live Application →](https://disasterlensss.streamlit.app/)**

The deployed application allows users to:

- Upload pre-disaster and post-disaster images.
- Compare the input images visually.
- Inspect a pixel-level difference visualization.
- Review model-predicted triage classes and their scores.
- Download a JSON report containing the analysis results.

## Key Features

- **Pre- and post-disaster image comparison:** Inspect satellite imagery captured at different times.
- **AI-based triage classification:** Generate scores for three response-prioritization classes.
- **Visual change inspection:** View image differences to help identify areas that may warrant further examination.
- **Probability-based outputs:** Review the model's scores for each supported class.
- **Downloadable JSON report:** Export structured inference results for documentation and further analysis.
- **Deep-learning inference:** Use a trained ResNet-18-based model implemented with PyTorch and TorchVision.
- **Interactive web application:** Access the system through a browser using Streamlit.
- **Reproducible project structure:** Keep application code, preprocessing utilities, model code, and tests organized in a Python package.

## How It Works

1. **Image input:** The user uploads a pre-event image and a post-event image.
2. **Preprocessing:** The images are converted and normalized according to the model's input requirements.
3. **Model inference:** The trained PyTorch model processes the input and produces scores for the supported triage classes.
4. **Visual comparison:** The application displays the input images and a pixel-level difference visualization.
5. **Result interpretation:** The user reviews the predicted action and associated class scores.
6. **Report generation:** A JSON report can be downloaded for further inspection.

> **Important:** Pixel-level differences do not necessarily represent physical damage. Differences in illumination, shadows, image alignment, sensor characteristics, seasonal conditions, and resolution may influence the output.

## Triage Categories

The current model predicts one of three classes:

| Class | Description |
|---|---|
| `discard` | Lower-priority imagery under the model's learned triage scheme. |
| `defer` | Imagery that may be reviewed later. |
| `downlink_now` | Imagery assigned higher priority for immediate transfer or review. |

These categories represent the model's learned labels. They are not official emergency-response recommendations, and their reliability depends on the quality and representativeness of the training data.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and model implementation |
| Streamlit | Interactive web interface |
| PyTorch | Deep-learning inference |
| TorchVision | ResNet-18 architecture and image transformations |
| Pillow | Image loading and manipulation |
| NumPy | Numerical operations |
| OpenCV | Image-processing utilities in the project package |
| pytest | Automated testing |
| Streamlit Community Cloud | Application deployment |
| Git and GitHub | Version control and source-code hosting |

## Model and Evaluation

The current best reported checkpoint is:

`artifacts/triage_gpu_v1.pt`

The checkpoint was evaluated on an event-held-out test set.

| Evaluation Metric | Reported Result |
|---|---:|
| Accuracy | 48.35% |
| Macro F1-score | 0.4501 |
| Evaluation examples | 91 |
| Training examples | 231 |
| Total triage examples | 322 |
| Event families | 17 |

### Understanding the Results

The dataset contains 322 examples divided into 231 training examples and 91 event-held-out evaluation examples. The split is organized by event family to evaluate performance on events excluded from training.

The current evaluation results indicate that the model remains experimental. Its predictions should not be interpreted as validated measurements of real-world disaster damage.

The dataset's labels were generated algorithmically using mask thresholds and have not been expert-reviewed. Consequently, the reported metrics measure agreement with the available labels, not necessarily agreement with verified physical damage or operational response priorities.

## Repository Structure

```text
DisasterLens/
├── app.py
├── artifacts/
│   └── triage_gpu_v1.pt
├── src/
│   └── disasterlens/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── labels.py
│       │   └── triage_dataset.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── metadata_baseline.py
│       │   ├── train.py
│       │   └── triage_model.py
│       └── preprocessing/
│           ├── __init__.py
│           └── tiling.py
├── tests/
│   ├── test_labels.py
│   └── test_tiling.py
├── requirements.txt
├── pyproject.toml
├── conftest.py
└── README.md
```

The repository also contains test files at the project root. Training and evaluation data are not shown in this tracked project structure.

## Getting Started

Follow these steps to run DisasterLens locally.

### Prerequisites

- Python compatible with the pinned dependencies. Python 3.12 is recommended for the current PyTorch 2.6.0 CPU requirements.
- Git
- An internet connection to install the dependencies

### 1. Clone the repository

```bash
git clone https://github.com/BharghavaIrigela/DisasterLens.git
cd DisasterLens
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The deployment requirements use CPU-only PyTorch packages. Make sure the selected Python version has compatible wheels for the pinned PyTorch and TorchVision versions.

### 4. Run the application

```bash
streamlit run app.py
```

After startup, Streamlit will display a local URL, typically:

`http://localhost:8501`

Open that address in your browser to use the application.

## Running Tests

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

The tests cover project functionality such as label handling and image tiling.

## Deployment

DisasterLens is deployed using **Streamlit Community Cloud**.

- **Application entry point:** `app.py`
- **Dependency configuration:** `requirements.txt`
- **Trained model:** `artifacts/triage_gpu_v1.pt`
- **Source repository:** [GitHub](https://github.com/BharghavaIrigela/DisasterLens)
- **Live deployment:** [disasterlensss.streamlit.app](https://disasterlensss.streamlit.app/)

To deploy your own copy, connect the GitHub repository to Streamlit Community Cloud, select the `main` branch, and specify `app.py` as the main application file. Choose a Python version compatible with the project's pinned dependencies.

## Responsible Use and Limitations

DisasterLens is intended for research, educational demonstrations, and decision-support experimentation. It is not a certified disaster-damage assessment system.

Important limitations include:

- The current event-held-out evaluation accuracy is 48.35%.
- Training labels were generated algorithmically and have not been expert-reviewed.
- Pixel differences can arise from factors unrelated to disaster damage.
- Generalization to unseen disasters, satellite sensors, geographical regions, and acquisition conditions has not been established.
- The model's scores should not be treated as calibrated probabilities of actual damage or as authoritative action recommendations.

**Do not use DisasterLens as the sole basis for emergency decisions, resource allocation, evacuation planning, or other safety-critical actions.** Operational use would require expert-annotated datasets, independent real-world validation, robust geospatial alignment, uncertainty analysis, and qualified human oversight.

## Future Improvements

Potential directions for further development include:

- Expanding evaluation to larger, independently sourced disaster datasets.
- Improving generalization across event types and satellite sensors.
- Incorporating more robust geospatial alignment and change-detection methods.
- Adding uncertainty estimation and confidence calibration.
- Evaluating performance using expert-verified damage labels.
- Improving reproducibility and documenting dataset provenance.
- Expanding the reporting interface with richer visual and geospatial outputs.

## Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Submit a pull request describing the improvement.

Repository: [DisasterLens on GitHub](https://github.com/BharghavaIrigela/DisasterLens)

## Data and Licensing

The triage dataset used during development is identified as **CC BY-NC 4.0**. Review the original dataset's license and attribution requirements before redistributing or reusing the data.

The dataset's license does not automatically establish the license for this repository's source code or trained model weights. Check those separately before redistribution or commercial use.

## Acknowledgements

DisasterLens builds on the open-source Python, Streamlit, PyTorch, TorchVision, Pillow, and related scientific-computing ecosystems.

---

<p align="center">
  <strong><a href="https://disasterlensss.streamlit.app/">🚀 Launch DisasterLens</a></strong>
  <br><br>
  <a href="https://github.com/BharghavaIrigela/DisasterLens">View Source Code on GitHub</a>
  <br><br>
  <em>A research prototype exploring satellite-based disaster assessment and response prioritization.</em>
</p>
