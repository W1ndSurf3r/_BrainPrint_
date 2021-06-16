import numpy as np
import torch

import torch as th


def evaluate_batch(_set, net, iterator, device = 'cuda:0'):
    total = 0
    correct = 0
    with th.no_grad():
        for inputs,targets in iterator.get_batches(_set, shuffle= True):
            input_vars = th.from_numpy(inputs).float().to(device)
            target_vars = th.from_numpy(targets).long().to(device)
            outputs = net(input_vars)
            _,predicted = th.max(outputs.data,1)
            #print(predicted)
            #print(outputs.size(), target_vars.size())
            total += target_vars.size(0)
            correct += (predicted == target_vars).sum().item()
    #print((predicted == target_vars).sum().item())
    return correct/total

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 200
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.load_model = False
        self.eval = False
    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
        elif score < self.best_score:
            self.counter += 1
            
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                #print(self.early_stop)
                self.early_stop = True
                self.counter = 0
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(optimizer.state_dict(),'checkpoint_opt.pt')
        self.val_loss_min = val_loss
        self.eval = True