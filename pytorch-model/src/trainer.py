"""Trainer.

This module creates a class that is able to train the network using the MNIST
dataset.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from classifier import Classifier
from os import path, mkdir
from time import strftime, gmtime
from plotter import PlotIt


class Trainer:
    def __init__(self, batch_size_train, batch_size_test, epochs, lr, momentum,
                 weight_file):
        """Training class that can train the Classifier class.

        Args:
            batch_size_train (int): Size of each batch for training.
            batch_size_test (int): Size of each batch for testing.
            epochs (int): Number of epochs to run training.
            lr (float): Learning rate.
            momentum (float): Training momentum.
            weight_file (str): Name of the weights file.
        """
        print("Loaded trainer.")
        # First create the data loaders as well as the mnist datasets
        mnist_train = torchvision \
            .datasets.MNIST('/files/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1308,),
                                                     (0.3081,))]))

        # Debug prints
        print("Finished loading MNIST Training dataset.")
        mnist_test = torchvision \
            .datasets.MNIST('/files/', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1308,),
                                                     (0.3081,))]))

        print("Finished loading MNIST Testing dataset.")
        self.train_loader = data.DataLoader(mnist_train,
                                            batch_size=batch_size_train,
                                            shuffle=True)
        print("Finished initializing training data loader.")
        self.test_loader = data.DataLoader(mnist_test,
                                           batch_size=batch_size_test,
                                           shuffle=True)
        print("Finished initializing testing data loader.")

        # Initialize classifier network and send it to CUDA
        self.classifier = Classifier()
        self.classifier.cuda()

        print("Successfully initialized classifier.")

        # Create optimizer and loss
        self.optimizer = optim.SGD(self.classifier.parameters(),
                                   lr=lr,
                                   momentum=momentum)

        # Making loss and counter lists
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

        self.epochs = epochs

        # Save location
        file_path = path.dirname(path.abspath(__file__))
        self.weight_path = path.join(file_path, 'results',
                                     '{}.pth'.format(weight_file))
        self.loss_path = path.join(file_path, 'results',
                                   strftime("%Y_%m_%d_%H-%M-%S", gmtime()))

        # Create results dir if non-existent
        if not path.isdir(path.join(file_path, 'results')):
            mkdir(path.join(file_path, 'results'))

    def start(self):
        """Start training of the network."""
        # TODO Load state dict
        if path.isfile(self.weight_path):
            self.classifier.load_state_dict(torch.load(self.weight_path))
            print("Loaded weights.")
        else:
            print("No weights found. Initializing with empty parameters.")
        print("Starting network training.")
        for epoch in range(self.epochs):
            self._train(epoch)
            self._test()

        # Save the state dict
        torch.save(self.classifier.state_dict(), self.weight_path)

        # TODO Print and save plot
        train_loss_file = open(self.loss_path + "-train-loss.csv", 'a')
        test_loss_file = open(self.loss_path + "-test-loss.csv", 'a')
        for item in self.train_losses:
            train_loss_file.write("{}\n".format(item))
        for item in self.test_losses:
            test_loss_file.write("{}\n".format(item))

        train_loss_file.close()
        test_loss_file.close()

        PlotIt(self.loss_path + "-train-loss.csv")
        print("Training completed successfully.")


    def _train(self, epoch):
        """Trains the network."""
        self.classifier.train()
        len_train_data = len(self.train_loader.dataset)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # Transform to cuda
            data = data.cuda()
            target = target.cuda()

            # Run the network
            output = self.classifier(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # Print statistics

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len_train_data,
                    100. * batch_idx * len(data) / len_train_data, loss.item()))

                # Append loss and counter to list
                self.train_losses .append(loss.item())
                self.train_counter.append((batch_idx*64)
                                          + ((epoch - 1)
                                          * len_train_data))

    def _test(self):
        """Validates the classifier and prints statistics."""
        self.classifier.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # Send to cuda
                data = data.cuda()
                target = target.cuda()

                # Run the network.
                output = self.classifier(data)
                test_loss += F.nll_loss(output, target,
                                        size_average=False).item()
                predicted = output.data.max(1, keepdim=True)[1]
                correct += predicted.eq(target.data.view_as(predicted)).sum()

            len_test_data = len(self.test_loader.dataset)

            test_loss /= len_test_data
            self.test_losses.append(test_loss)
            print("Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)"
                  .format(test_loss, correct, len_test_data,
                          100. * correct / len_test_data))
