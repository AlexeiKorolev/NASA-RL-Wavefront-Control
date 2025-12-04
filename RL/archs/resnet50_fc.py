import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet50FC, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, output_dim)
        
        # If input_dim is different from 3 (RGB), add a convolutional layer to adjust input channels
        if input_dim != 3:
            self.input_adjust = nn.Conv2d(input_dim, 3, kernel_size=1)
        else:
            self.input_adjust = None

    def forward(self, x):
        # Resize input image dimensions to match ResNet-50 expected input size (3x224x224)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Adjust input channels if necessary
        if self.input_adjust:
            x = self.input_adjust(x)
        
        # Forward pass through ResNet-50
        x = self.resnet50(x)
        return x