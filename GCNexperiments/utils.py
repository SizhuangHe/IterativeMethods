import numpy as np
import scipy.sparse as sp
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def accuracy(guess, truth):
    correct = guess == truth
    acc = correct.sum().item() / truth.size(dim=0)
    return acc

def train(epoch, model, optimizer, data):
    t = time.time()
    
    model.train()
    output = model(data.x, data.edge_index)
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    pred = output[data.train_mask].argmax(dim=1)
    acc_train = accuracy(pred, data.y[data.train_mask])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(data.x, data.edge_index)

    loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    return loss_train, acc_train, loss_val, acc_val


def test(model, data):
    model.eval()
    
    start_time = time.time()
    output = model(data.x, data.edge_index)
    end_time = time.time()
    
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    pred = output[data.test_mask].argmax(dim=1)
    acc_test = accuracy(pred, data.y[data.test_mask])
    
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test))
    # print("Testing time: ", end_time-start_time)
    
    return loss_test, acc_test
    

def run_experiment(model, data, lr, weight_decay, model_name, run, num_epochs=200):
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

        output = model(data.x, data.edge_index)   
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        pred = output[data.train_mask].argmax(dim=1)
        acc_train = accuracy(pred, data.y[data.train_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        

        model.eval()
        output = model(data.x, data.edge_index)
        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        pred = output[data.val_mask].argmax(dim=1)
        acc_val = accuracy(pred, data.y[data.val_mask])
        
        loss_TRAIN.append(loss_train.item())
        acc_TRAIN.append(acc_train)
        loss_VAL.append(loss_val.item())
        acc_VAL.append(acc_val)
        
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val),
            'time: {:.4f}s'.format(time.time() - t))

    total_end = time.time()
    training_time = total_end - total_start
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(training_time))
    
    # Testing
    loss_test, acc_test = test(model, data)

    # title = model_name + "_run_" + str(run)
    # file_name = "Figures/" + title + ".png"
    # #Summary graphs
    # fig, ax = plt.subplots(2, 2)
    # fig.tight_layout(pad=5.0)
    # ax[0, 0].plot(loss_TRAIN, 'b') #row=0, col=0
    # ax[0, 0].title.set_text("Training loss")
    # ax[0, 1].plot(acc_TRAIN, 'b') #row=0, col=1
    # ax[0, 1].title.set_text("Training accuracy")
    # ax[1, 0].plot(loss_VAL, 'b') #row=1, col=0
    # ax[1, 0].title.set_text("Validation loss")
    # ax[1, 1].plot(acc_VAL, 'b') #row=1, col=1
    # ax[1, 1].title.set_text("Validation accuracy")
    # fig.suptitle(title)
    # fig.savefig(file_name)

    return loss_test.item(), acc_test, training_time

def print_stats(model_name, acc_test, training_time):
    print("Experiment statistics for ", model_name)
    mean_acc = np.mean(acc_test)
    std_acc = np.std(acc_test)
    mean_time = np.mean(training_time)
    print("Mean accuracy: ", mean_acc)
    print("std of accuracy: ", std_acc)
    print("Mean training time: ", mean_time)
