
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class TriageModel(nn.Module):
    """Paired-image triage classifier using pretrained ResNet-18."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Keep the pretrained convolutional feature extractor.
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        # Initially freeze the pretrained encoder for stable CPU training.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # Compare baseline and current images using their features.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = x[:, :3, :, :]
        current = x[:, 3:, :, :]

        baseline_features = self.encoder(baseline).flatten(1)
        current_features = self.encoder(current).flatten(1)

        difference = torch.abs(current_features - baseline_features)

        combined = torch.cat(
            [baseline_features, current_features, difference], dim=1
        )

        return self.classifier(combined)
