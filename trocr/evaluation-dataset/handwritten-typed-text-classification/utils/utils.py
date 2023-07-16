import numpy as np
from skimage import morphology, color

import torch
import torch.nn as nn

class PartialErosion:
    def __init__(self, iterations=2, selem=morphology.square(3)):
        self.iterations = iterations
        self.selem = selem

    def __call__(self, image):
        
        # Convert the image to a numpy array
        np_image = np.array(image)
        
        if isinstance(np_image, np.ndarray) and len(np_image.shape) == 3:
            # Create an empty array with same shape as input image
            result = np.zeros_like(np_image)
            # Apply erosion to each channel independently
            for channel in range(np_image.shape[2]):
                temp_image = np_image[:, :, channel]
                for i in range(self.iterations):
                    temp_image = morphology.erosion(temp_image, self.selem)
                result[:, :, channel] = temp_image
            return result
        else:
            raise ValueError("Input image must be a 3D numpy array (height x width x channels).")
    
    def __repr__(self):
        return f'PartialErosion(iterations={self.iterations}, selem={self.selem})'


class VGG16Binary(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super(VGG16Binary, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * (input_shape[1] // 8) * (input_shape[2] // 8), 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
    
    
