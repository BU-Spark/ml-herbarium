import numpy as np
from skimage import morphology, color

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import glob

# Interleaved CRNN type model defintion to train from scratch
class InterleavedCRNN(nn.Module):
    def __init__(self, num_classes):
        super(InterleavedCRNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rnn1 = nn.GRU(input_size=64*50, hidden_size=32, num_layers=1, bidirectional=True, dropout=0.5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rnn2 = nn.GRU(input_size=640, hidden_size=64, num_layers=1, bidirectional=True, dropout=0.5)
        
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # pass through first conv block
        x = self.conv1(x)
        
        # reshape for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, -1)

        # pass through first RNN block
        x, _ = self.rnn1(x)

        # reshape for conv block
        x = x.view(b, 10, c, 10).permute(0, 2, 3, 1).contiguous()

        # pass through second conv block
        x = self.conv2(x)

        # reshape for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, -1)

        # pass through second RNN block
        x, _ = self.rnn2(x)
        
        # reshape for fc
        x = x[:, -1, :]

        # pass through FC layer
        x = self.fc(x)

        return x


# CRNN model type model defintion to train from scratch
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.rnn = nn.GRU(input_size=128, hidden_size=256, num_layers=4, bidirectional=False, dropout=0.3)
        self.fc = nn.Linear(256, num_classes)  # Adjusted input size for the linear layer

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # Reshape for RNN
        b, w, h, c = x.size()
        x = x.view(b, -1, c).permute(1, 0, 2)  # merge height and channel dimensions
        _, hidden = self.rnn(x)
        x = hidden[-1]
        x = self.fc(x)
        return x  # Adjusted input size for the linear layer

# Custom image transform that applies a specified #iteration of the erosion morphological operation
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

# VGG16 type model defintion to train from scratch
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


# Data loader that applies transforms from TrOCRProcessor
class TrOCRPreprocessor(ImageFolder):
    def __init__(self, root, processor, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.processor = processor

    def __getitem__(self, index):
        # We retrieve the image and label like in the default __getitem__ method
        img, target = super().__getitem__(index)

        # We apply the TrOCR processor to the image
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values

        # We return the tensor under the "pixel_values" key along with the target
        return pixel_values, target


# Dataset to load the stored tensors from the TrOCR encoder
class TensorDataset(Dataset):
    def __init__(self, tensor_directory):
        self.tensor_files = glob.glob(f'{tensor_directory}/*image_representation*.pt')
        self.label_files = glob.glob(f'{tensor_directory}/labels*.pt')
        self.tensor_files.sort()  # make sure that the tensor files are in the correct order
        self.label_files.sort()  # make sure that the tensor files are in the correct order
        
    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_file = self.tensor_files[idx]
        lable_file = self.label_files[idx]
        tensor = torch.load(tensor_file)
        labels = torch.load(lable_file)
        
        return tensor, labels
