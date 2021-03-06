"""Classifier.

This module will serve as the main classifier network for the MNIST Dataset.
The structure of the network will be simple.
    Conv2D -> Conv2D with dropout -> Fully Connected -> Fully connected -> Out

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        """Classifier network designed to classify handwritten numbers."""
        # As always call super's init
        super(Classifier, self).__init__()

        # Now we define the layers to be used.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            # nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(2000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def num_flat_features(x):
        """Multiplies the number of features for flattening a convolution.
        References:
            https://pytorch.org/tutorials/beginner/blitz/neural_networks_
            tutorial.html
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
