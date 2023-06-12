import numpy as np
import scipy.sparse as sp
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    

    return adj, features, labels,


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# added here by Sizhuang
def train(epoch, model, optimizer, features, adj, idx_train, idx_val, labels):
    t = time.time()
    
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, idx_test, labels):
    model.eval()
    
    start_time = time.time()
    output = model(features, adj)
    end_time = time.time()
    
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("Testing time: ", end_time-start_time)
    
    return loss_test, acc_test
    

def run_experiment(num_epochs, model, lr, weight_decay, features, adj, idx_train, idx_val, idx_test, labels, model_name, run):
    print("runrunrun!")

    # current_dir = os.path.dirname(__file__)
    # relative_path = save_path
    # abs_path = os.path.join()

    loss_TRAIN = []
    acc_TRAIN = []
    loss_VAL = []
    acc_VAL = []

    optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)
    total_start = time.time()
    for epoch in range(num_epochs):
        t = time.time()
    
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_TRAIN.append(loss_train.item())
        acc_TRAIN.append(acc_train.item())

        loss_train.backward()
        optimizer.step()

        
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_VAL.append(loss_val.item())
        acc_VAL.append(acc_val.item())
    
    total_end = time.time()
    training_time = total_end - total_start
    

        # print('Epoch: {:04d}'.format(epoch+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()),
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))
        

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(training_time))
    
    # Testing
    loss_test, acc_test = test(model, features, adj, idx_test, labels)

    title = model_name + "_run_" + str(run)
    file_name = "Figures/" + title + ".png"
    #Summary graphs
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=5.0)
    ax[0, 0].plot(loss_TRAIN, 'b') #row=0, col=0
    ax[0, 0].title.set_text("Training loss")
    ax[0, 1].plot(acc_TRAIN, 'b') #row=0, col=1
    ax[0, 1].title.set_text("Training accuracy")
    ax[1, 0].plot(loss_VAL, 'b') #row=1, col=0
    ax[1, 0].title.set_text("Validation loss")
    ax[1, 1].plot(acc_VAL, 'b') #row=1, col=1
    ax[1, 1].title.set_text("Validation accuracy")
    fig.suptitle(title)
    fig.savefig(file_name)

    return loss_test, acc_test, training_time

def print_stats(model_name, acc_test, training_time):
    print("Experiment statistics for ", model_name)
    mean_acc = np.mean(acc_test)
    std_acc = np.std(acc_test)
    mean_time = np.mean(training_time)
    print("Mean accuracy: ", mean_acc)
    print("std of accuracy: ", std_acc)
    print("Mean training time: ", mean_time)
