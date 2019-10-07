import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='sbu')
parser.add_argument('--seed', type=int, metavar='N',
                    help='random number generator seed, use "-1" for random seed',
                    default=1570254494)
parser.add_argument('--num-synth', type=int, metavar='N',
                    help='number of synthetic samples to generate',
                    default=1)
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)


# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda


# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Load the dataset
    log.set_dataset_name(args.dataset)
    dataset = DataFactory.instantiate(args.dataset, args.num_synth)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)

    # Run each fold and average the results
    accuracies = []

    for fold_idx in range(dataset.num_folds):
        log('Running fold "{}"...'.format(fold_idx))

        test_accuracy = run_fold(dataset, fold_idx, use_cuda)
        accuracies += [test_accuracy]

        log('Fold "{}" complete, final accuracy: {}'.format(fold_idx, test_accuracy))

    log('')
    log('-----------------------------------------------------------------------')
    log('Training complete!')
    log('Average accuracy: {}'.format(np.mean(accuracies)))


# ----------------------------------------------------------------------------------------------------------------------
def run_fold(dataset, fold_idx, use_cuda):
    """
    Trains/tests the model on the given fold
    """

    hyperparameters = dataset.get_hyperparameter_set()
    # Instantiate the model, loss measure and optimizer
    model = DeepGRU(dataset.num_features, dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparameters.learning_rate,
                                 weight_decay=hyperparameters.weight_decay)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         shuffle=True,
                                                         random_seed=seed+fold_idx,
                                                         normalize=True)

    best_train_accuracy = 0
    best_test_accuracy = 0

    # Train the model
    for epoch in range(hyperparameters.num_epochs):
        loss_meter = AverageMeter()
        train_meter = AverageMeter()
        test_meter = AverageMeter()

        #
        # Training loop
        #
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update stats
            loss_meter.update(loss.item(), curr_batch_size)
            train_meter.update(accuracy, curr_batch_size)

        train_accuracy = train_meter.avg

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        log('Epoch: [{0}]'.format(epoch))
        log('       [Avg Loss]          {loss.avg:.6f}'.format(loss=loss_meter))
        log('       [Training]   Prec@1 {top1.avg:.6f} Max {max:.6f}'
             .format(top1=train_meter, max=best_train_accuracy))

        #
        # Testing loop
        #
        model.eval()
        with torch.no_grad():
            test_loss_meter = AverageMeter()

            for batch in test_loader:

                accuracy, curr_batch_size, loss = run_batch(batch, model, criterion)
                test_loss_meter.update(loss.item(), curr_batch_size)
                test_meter.update(accuracy, curr_batch_size)

            test_accuracy = test_meter.avg

            # Update best accuracies
            if best_test_accuracy < test_accuracy:
                best_test_accuracy = test_accuracy

            log('       [Avg Loss]          {loss.avg:.6f}'.format(loss=test_loss_meter))
            log('       [Validation] Prec@1 {top1:.6f} Max {max:.6f}'
                 .format(top1=test_accuracy, max=best_test_accuracy))

        if loss_meter.avg <= 1e-6 or best_test_accuracy == 100:
            break

    return best_test_accuracy


# ----------------------------------------------------------------------------------------------------------------------
def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch

    if use_cuda:
        examples = examples.cuda()
        labels = labels.cuda()

    # Forward and loss computation
    outputs = model(examples, lengths)
    loss = criterion(outputs, labels)

    # Compute the accuracy
    predicted = outputs.argmax(1)
    correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    accuracy = correct / curr_batch_size * 100.0

    return accuracy, curr_batch_size, loss


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
