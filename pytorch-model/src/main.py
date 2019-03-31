"""Main.

Main file that runs everything else. Can be used to either train or run the
network. It run using command line arguments.
"""
import argparse
from trainer import Trainer


# Parse arguments
def parse_arguments():
    """Parses command line arguments."""
    description = "Runs a classifier on the MNIST Dataset."
    parser = argparse.ArgumentParser(description=description)

    # Training arguments
    parser.add_argument('-t', '--train', action='store_true',
                        help="set to training mode")
    parser.add_argument('-g', '--batch-size-train', type=int, nargs='?',
                        default=64,
                        help="size of training batch")
    parser.add_argument('-p', '--epochs', type=int, nargs='?',
                        default=1,
                        help="number of epochs to train for")
    parser.add_argument('-j', '--batch-size-test', type=int, nargs='?',
                        default=1000,
                        help='size of the test batch')
    parser.add_argument('-l', '--lr', type=float, nargs='?',
                        default=0.01,
                        help="learning rate")
    parser.add_argument('-m', '--momentum', type=float, nargs='?',
                        default=0.5,
                        help="training momentum")
    parser.add_argument('-w', '--weights', type=str, nargs='?',
                        default='weights',
                        help="file name of the weights file.")

    # Evaluation
    # Not yet implemented

    return parser.parse_args()


def main(arguments):
    """Main function that actually runs everything."""
    if arguments.t:
        # Runs in training mode
        trainer = Trainer(arguments.batch_size_train, arguments.batch_size_test,
                          arguments.epochs, arguments.lr, arguments.momentum,
                          arguments.weights)

        trainer.start()
