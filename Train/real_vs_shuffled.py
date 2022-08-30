# Imports
import os
import argparse
import numpy as np
from collections import Counter
import torch
from torch import nn, optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Constants
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()


class ClassifyNetwork(nn.Module):
    """
    A class used to represent a feed-forward neural network classifier

    Attributes
    ----------
    layers_dims : list
        The dimensions of the neural network
    dropout_rate : float
        The dropout rate to be used in the network

    Methods
    -------
    forward(x)
        Forward pass of the neural network on input x
    """
    def __init__(self, layers_dims, dropout_rate):
        super(ClassifyNetwork, self).__init__()

        # 1st layer + ReLU activation function
        self.linear = nn.Linear(layers_dims[0], layers_dims[1])
        self.act1 = nn.ReLU()

        # 2nd layer + ReLU activation function
        self.linear2 = nn.Linear(layers_dims[1], layers_dims[2])
        self.act2 = nn.ReLU()

        # 3rd layer + Sigmoid activation function
        self.linear3 = nn.Linear(layers_dims[2], 1)
        self.act3 = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass of the neural network on input x

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape NxM where N equals the batch size
            and M equals the number of features

        Returns
        -------
        torch.Tensor
            Classification of each example from the input data, shape N
        """
        x = self.act1(self.linear(x))
        x = self.dropout(x)
        x = self.act2(self.linear2(x))
        x = self.dropout(x)
        x = self.act3(self.linear3(x))
        return x.squeeze()

# Helper Functions


def initialize_weights(model):
    """Initialize neural network weights using xavier method,
    also initializes the biases of the network to zero

    Parameters
    ----------
    model : ClassifyNetwork instance
        The neural network weights and biases to be updated
    """
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(model.bias)


def compute_metrics(labels, predictions):
    """Computes evaluation matrices of the model for the validation step

    Parameters
    ----------
    labels : torch.Tensor
        Real labels of the data

    predictions : torch.Tensor
        Model classifications of the data

    Returns
    -------
    dict
        The following measurements on the validation data: Accuracy, F1,
        Precision, Recall & AUC
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC': auroc
    }


def train_model(model, data_loader, optim):
    """Training mode of the neural network

    Parameters
    ----------
    model : ClassifyNetwork instance
        Represents a feed-forward neural network classifier

    data_loader : torch.utils.data.DataLoader
        Load data into the model via batches

    optim : torch.optim
        Optimizer of the model

    Returns
    -------
    loss : float
        Training loss of the model

    accuracy : float
        Training accuracy of the model
    """
    model.train()
    running_loss = 0.0
    correct = 0
    for batch_num, (inputs, labels) in enumerate(data_loader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = (output > 0.5).float()
        correct += (output == labels).float().sum()

    accuracy = 100 * correct / len(data_loader.dataset)
    loss = running_loss / batch_num
    print(f"Train loss: {loss:.3f}, Train accuracy: {accuracy:.3f}%")
    return loss, accuracy.item()


def eval_model(model, data_loader):
    """Validation mode of the neural network

    Parameters
    ----------

    model : ClassifyNetwork instance
        Represents a feed-forward neural network classifier

    data_loader : torch.utils.data.DataLoader
        Load data into the model via batches

    Returns
    -------
    loss : float
        Validation loss of the model

    metrics : dict
        The following measurements on the validation data: Accuracy, F1,
        Precision, Recall & AUC
    """
    losses = []
    all_outputs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            all_outputs.extend(torch.round(output).cpu())
            all_labels.extend(labels.cpu())
            loss = criterion(output, labels)
            losses.append(loss.item())

        metrics = compute_metrics(all_labels, all_outputs)
        print(f"Validation loss: {np.mean(losses):.3f}")
        print(f"Validation metrics: {metrics}")
    return np.mean(losses), metrics


def load_and_reshape_data(embeddings_path):
    """Load proteins sequences embeddings and reshape them,
    To fit as input for the neural network

    Parameters
    ----------
    embeddings_path : str
        Absolute path of the proteins sequences embeddings from DistilProtBert model

    Returns
    -------
        sequences : torch.Tensor of shape 512*64
            embeddings in the suitable shape for the neural network classifier,
            (512 is the maximum protein sequence length & 64 is the representation of
            each amino acid after max pooling)
    """
    sequences = torch.load(embeddings_path).float()
    return sequences.view(sequences.shape[0], sequences.shape[1] * sequences.shape[2])


def load_data(natural_path, shuffled_path):
    """Load the natural proteins sequences embeddings and the shuffled ones

    Parameters
    ----------
    natural_path : str
        Absolute path of the natural proteins sequences embeddings from DistilProtBert model

    shuffled_path : str
        Absolute path of the shuffled proteins sequences embeddings from DistilProtBert model

    Returns
    -------
    (natural proteins, shuffled proteins) : (torch.Tensor of shape 512*64, torch.Tensor of shape 512*64)
        natural / shuffled protein sequences embeddings in the suitable shape for the neural network classifier,
        (512 is the maximum protein sequence length & 64 is the representation of each amino acid after max pooling)
    """
    natural_proteins = load_and_reshape_data(natural_path)
    shuffled_sequences = load_and_reshape_data(shuffled_path)
    return natural_proteins, shuffled_sequences


def get_fold(x_chunks, y_chunks, folds, current_split):
    x_fold = torch.vstack([x_chunks[j] for j in range(folds) if j != current_split])
    y_fold = torch.hstack([y_chunks[j] for j in range(folds) if j != current_split])
    return x_fold, y_fold


def shuffle_fold(x_fold, y_fold):
    fold_perm = torch.randperm(x_fold.shape[0])
    return x_fold[fold_perm], y_fold[fold_perm]


def get_folds(x_n_chunks, x_s_chunks, y_n_chunks, y_s_chunks, folds):
    all_folds = []

    for i in range(folds):
        x_fold_train_n, y_fold_train_n = get_fold(x_n_chunks, y_n_chunks, folds, i)
        x_fold_train_s, y_fold_train_s = get_fold(x_s_chunks, y_s_chunks, folds, i)

        x_fold_train = torch.cat((x_fold_train_n, x_fold_train_s))
        y_fold_train = torch.cat((y_fold_train_n, y_fold_train_s))

        # shuffle training set fold
        x_full_train_fold, y_full_train_fold = shuffle_fold(x_fold_train, y_fold_train)

        x_full_val_fold = torch.cat((x_n_chunks[i], x_s_chunks[i]))
        y_full_val_fold = torch.cat((y_n_chunks[i], y_s_chunks[i]))

        # shuffle validation set fold
        x_full_val_fold, y_full_val_fold = shuffle_fold(x_full_val_fold, y_full_val_fold)

        all_folds.append({'x_train': x_full_train_fold, 'x_val': x_full_val_fold,
                          'y_train': y_full_train_fold, 'y_val': y_full_val_fold})
    return all_folds


def split_data(natural_proteins_embeddings_path, shuffled_sequences_embeddings_path, folds):
    x_train_natural, x_train_shuffled = load_data(natural_proteins_embeddings_path, shuffled_sequences_embeddings_path)
    y_train_ones = torch.ones((len(x_train_natural)))
    y_train_zeros = torch.zeros((len(x_train_shuffled)))

    chunk_size = len(x_train_natural) // 10

    x_natural_chunks = torch.split(x_train_natural, chunk_size)
    x_shuffled_chunks = torch.split(x_train_shuffled, chunk_size)

    y_natural_chunks = torch.split(y_train_ones, chunk_size)
    y_shuffled_chunks = torch.split(y_train_zeros, chunk_size)

    return get_folds(x_natural_chunks, x_shuffled_chunks, y_natural_chunks, y_shuffled_chunks, folds)


def trim_zeros_from_metrics(folds_num, metrics_dic):
    metrics_trimmed_dic = {i: 0 for i in range(folds_num)}
    for i, epochs_metric in enumerate(metrics_dic):
        metrics_trimmed_dic[i] = np.trim_zeros(epochs_metric, trim='b')
    return metrics_trimmed_dic


def get_final_metric_from_each_fold(metric_folds_dic):
    return [metric_folds_dic[fold_i][-1] for fold_i in metric_folds_dic]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilProtBert implementation')
    parser.add_argument('--output_dir', default='./', help='output dir to save the model')
    parser.add_argument('--n_folds', default=10, type=int, help='Number of train folds. Default is 10.')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of train epochs. Default is 100.')
    parser.add_argument('--lr', default=0.000001, type=float, help='learning rate. Default is 0.000001')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Dropout rate. Default is 0.1')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size. Default is 32.')
    parser.add_argument("--nn_dimensions", nargs="+", default=[512*64, 512*64//32, 512*64//128],
                        help='layers dimensions of the neural network. Default is: 512*64, 512*64/32 and 512*64/128')
    parser.add_argument('--use_early_stopping', default=True, help='train with early stopping. Default is True.')
    parser.add_argument('--tensorboard_log_dir', default='./', help='tensorboard logs directory.')
    parser.add_argument('--proteins_embeddings_file_path', default='./',
                        help='natural proteins features extracted using DistilProtBert.')
    parser.add_argument('--shuffled_sequences_file_path', default='./',
                        help='shuffled sequences features extracted using DistilProtBert.')
    args = parser.parse_args()

    num_epochs = args.n_epochs
    num_folds = args.n_folds

    data_folds = split_data(args.proteins_embeddings_file_path, args.shuffled_sequences_file_path, num_folds)

    # log results
    results_train_folds = {'loss': np.zeros(shape=(num_folds, num_epochs)),
                           'acc': np.zeros(shape=(num_folds, num_epochs))}
    eval_folds = {'loss': np.zeros(shape=(num_folds, num_epochs)), 'Accuracy': np.zeros(shape=(num_folds, num_epochs)),
                  'F1': np.zeros(shape=(num_folds, num_epochs)), 'Precision': np.zeros(shape=(num_folds, num_epochs)),
                  'Recall': np.zeros(shape=(num_folds, num_epochs)), 'AUC': np.zeros(shape=(num_folds, num_epochs))}

    for fold, fold_data in enumerate(data_folds):
        print()
        print(f'fold num: {fold + 1}')

        fold_log_dir = args.tensorboard_log_dir+f'{fold}'
        if not os.path.exists(fold_log_dir):
            os.makedirs(fold_log_dir)

        writer = SummaryWriter(log_dir=fold_log_dir)

        # split data to train & validation
        X_train, X_val = fold_data['x_train'], fold_data['x_val']
        y_train, y_val = fold_data['y_train'], fold_data['y_val']

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

        # init model
        cls_nn = ClassifyNetwork(args.nn_dimensions, args.dropout_rate)
        print('classifier network architecture:')
        summary(cls_nn)
        cls_nn.apply(initialize_weights)
        cls_nn.to(device)
        optimizer = optim.RAdam(cls_nn.parameters(), lr=args.lr, eps=1e-08)

        # early stopping params
        curr_best_val_loss = 10.0
        patience = 2

        # train model
        for epoch in range(num_epochs):
            print(f'epoch number: {epoch + 1}')
            train_loss, train_acc = train_model(cls_nn, train_loader, optimizer)
            results_train_folds['loss'][fold, epoch] = train_loss
            results_train_folds['acc'][fold, epoch] = train_acc

            # evaluate model
            val_loss, val_metrics = eval_model(cls_nn, val_loader)
            eval_folds['loss'][fold, epoch] = val_loss
            eval_folds['Accuracy'][fold, epoch] = val_metrics['Accuracy']
            eval_folds['F1'][fold, epoch] = val_metrics['F1']
            eval_folds['Precision'][fold, epoch] = val_metrics['Precision']
            eval_folds['Recall'][fold, epoch] = val_metrics['Recall']
            eval_folds['AUC'][fold, epoch] = val_metrics['AUC']

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/eval", val_loss, epoch)
            writer.add_scalar("Accuracy/eval", val_metrics['Accuracy'], epoch)
            writer.add_scalar("F1/eval", val_metrics['F1'], epoch)
            writer.add_scalar("Precision/eval", val_metrics['Precision'], epoch)
            writer.add_scalar("Recall/eval", val_metrics['Recall'], epoch)
            writer.add_scalar("AUC/eval", val_metrics['AUC'], epoch)

            # check early stopping criterion
            if args.use_early_stopping:
                if val_loss <= curr_best_val_loss:
                    curr_best_val_loss = val_loss
                    patience = 2
                else:
                    patience -= 1

                if patience == 0:
                    break

        # save model (per fold) state dict
        torch.save(cls_nn.state_dict(), args.output_dir+f'model_fold_{fold}.pt')

        writer.flush()
        writer.close()

    eval_loss = trim_zeros_from_metrics(num_folds, eval_folds['loss'])
    eval_acc = trim_zeros_from_metrics(num_folds, eval_folds['Accuracy'])
    eval_f1 = trim_zeros_from_metrics(num_folds, eval_folds['F1'])
    eval_precision = trim_zeros_from_metrics(num_folds, eval_folds['Precision'])
    eval_recall = trim_zeros_from_metrics(num_folds, eval_folds['Recall'])
    eval_auc = trim_zeros_from_metrics(num_folds, eval_folds['AUC'])

    loss = get_final_metric_from_each_fold(eval_loss)
    acc = get_final_metric_from_each_fold(eval_acc)
    f1 = get_final_metric_from_each_fold(eval_f1)
    precision = get_final_metric_from_each_fold(eval_precision)
    recall = get_final_metric_from_each_fold(eval_recall)
    auc = get_final_metric_from_each_fold(eval_auc)

    print()
    print('avg loss:', np.average(loss))
    print('avg acc:', np.average(acc))
    print('avg f1:', np.average(f1))
    print('avg precision:', np.average(precision))
    print('avg recall:', np.average(recall))
    print('avg auc:', np.average(auc))
    print(f'best auc is: {max(auc)}, fold number: {np.argmax(auc)}')
