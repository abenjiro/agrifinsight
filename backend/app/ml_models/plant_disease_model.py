import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights

class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b3', pretrained=True, dropout_rate=0.3):
        super(PlantDiseaseClassifier, self).__init__()

        self.model_name = model_name

        if model_name == 'efficientnet_b3':
            if pretrained:
                weights = EfficientNet_B3_Weights.DEFAULT
                self.backbone = models.efficientnet_b3(weights=weights)
            else:
                self.backbone = models.efficientnet_b3(weights=None)

            # Get the number of input features for the classifier
            num_features = self.backbone.classifier[1].in_features

            # Replace classifier with custom head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)

            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'mobilenet_v3':
            if pretrained:
                self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            else:
                self.backbone = models.mobilenet_v3_large(weights=None)

            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.Hardswish(),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)
