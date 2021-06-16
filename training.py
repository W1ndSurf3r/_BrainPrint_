import sys
# sys.path.append('../')
import os
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import StepLR , ExponentialLR
import numpy as np
import math, random
import matplotlib.pyplot as plt

from datautils.iterators import BalancedBatchSizeIterator
from datautils.splitters import split_into_two_sets, concatenate_sets
from Models.pytorchtools import EarlyStopping, evaluate_batch
from braindecode.datautil.signal_target import SignalAndTarget
from utils.utils import conc_augmented, MaxNormDefaultConstraint, crop

### Available nets###
from Models.Conv_Rnn import convrnn
from Models.DemixingGRU import DemixGRU

##############

def example():
    acc = list()
    for i in range(1,10,1):

        #net = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
        train_set, test_set = load_data(i)
        train_set, test_set = crop(train_set, test_set)
        train_set, valid_set, test_set = prepare_data(train_set, test_set, one_hot= False)
        #train_set, valid_set, test_set = format_for_rnn(train_set, valid_set, test_set)
        print(train_set.X.shape)
        #net = DRNN(n_input = input_dim, n_hidden = 10, n_layers= 3, dropout=0, cell_type='GRU', batch_first=True)
        #net = CONVDRNN()
        net= convrnn()
        model = Network(model=net,batch_size=32)
        test_acc  = model.train(train_set, valid_set,test_set,n_epochs = 500)
        acc.append(test_acc)
    return acc
def load_data(subject_id, datasetName = '../bciiv2b',  ival = None):
    """
        Load pkl file.
        # Subject id: Integer of the subject id
        # datasetName: The folder containing the data
        # hidden_size: dimension of the hidden state
    
    
    """
    
    dataset_path =  str(subject_id) +'_'+ 'set.pkl'
    dataset_path = os.path.join( datasetName,dataset_path)
    with open(dataset_path, 'rb') as handle:
            dataset = pickle.load(handle)
    
    
    train_set = dataset['train_set']
    test_set  = dataset['test_set']
    train_set = conc_augmented(train_set)
    if ival != None:
        train_set.X = train_set.X[:,:,ival[0]: ival[1]]
        test_set.X = test_set.X[:,:,ival[0]: ival[1]]
    else:
        train_set.X = train_set.X
        test_set.X = test_set.X
    train_set.y = train_set.y
    print('Loaded: {}'.format(dataset_path))
    return train_set, test_set
def format_for_rnn(*args):
    sets = list()
    for set_ in args:
        x = set_.X
        x = x.reshape(x.shape[0], x.shape[1], 5, x.shape[2]//5)
        #set_.X = x[:,0]
        sets.append(set_)
    return sets
def prepare_data(train_set,test_set, val_percentatge= 0.01, one_hot = False):
    if one_hot:
        nb_classes = len(np.unique(train_set.y))
        train_set.y = convert_to_one_hot(nb_classes, train_set.y)
        test_set.y = convert_to_one_hot(nb_classes,test_set.y)
    else:
        
        train_set.y = np.expand_dims(train_set.y,1)
        test_set.y =  np.expand_dims(test_set.y,1)
        print(train_set.y.shape)
    train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=1 - val_percentatge) 
    print('Train set: {}, Valid set: {}, Test set: {}'.format(len(train_set.X),len(valid_set.X), len(test_set.X)))
    return train_set, valid_set, test_set

def convert_to_one_hot(nb_classes, labels):
    targets = np.array(labels).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets
def evaluate(model, test_set, train_iterator, pred = False):
    model.eval()
    outputs = []
    targets = []
    total = 0
    correct = 0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for inputs,targets in train_iterator.get_batches(test_set, shuffle= True):
        input_vars = torch.from_numpy(inputs).float().to(device)
        target_vars = torch.from_numpy(targets).long().to(device)
        target_vars = Variable(target_vars.view(-1))
        #print(input_vars.size())
       
        out, alphas, spat_attn = model(input_vars.to(device).float())
        _,predicted = torch.max(out.data,1)
        outputs.append(predicted.detach().cpu().numpy().flatten())
        correct_pred = torch.eq(target_vars, predicted)
        #print(predicted[0],target_vars[0])
        #print(torch.sum(correct_pred))
        correct += torch.sum(correct_pred).item()
        total += len(target_vars)
        #print(correct)
        
    #print(correct/total)
    if pred:
        input_vars = torch.from_numpy(test_set.X).float().to(device)
        target_vars = torch.from_numpy(test_set.y).long().to(device)
        out, alphas,spat_attn = model(input_vars.to(device).float())
        _,predicted = torch.max(out.data,1)
        predicted = predicted.detach().cpu().numpy()
        return correct/total, predicted
    else:
        return correct/total



class Network(object):
    def __init__(self, model= None, criterion = None, optimizer= None, batch_size = 32, early_stopping = True):
        
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.iterator = BalancedBatchSizeIterator(batch_size=batch_size)
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.model_constraint = MaxNormDefaultConstraint()
          #self.__dict__.update(locals())
        if early_stopping:
            self.early_stopping = EarlyStopping(verbose= True)
        
    def train_one_epoch(self,train_set):
        self.model.train()
        avg_loss = 0.
        
        for inputs, targets in self.iterator.get_batches(train_set, shuffle= True):
            input_vars = torch.from_numpy(inputs).float().to(self.device)
            target_vars = torch.from_numpy(targets).long().to(self.device)
            target_vars = Variable(target_vars.view(-1))
            #print(input_vars.size())
            #model.zero_grad()
            self.optimizer.zero_grad()
            
            
            out, alphas, spat_attn = self.model(input_vars)
            #print(target_vars.size())
            loss = self.criterion(out, target_vars)
            loss.backward()
            self.optimizer.step()
            #self.model_constraint.apply(self.model)
            avg_loss += loss.item()
        return avg_loss/ len(train_set.X), alphas,spat_attn
          
    def evaluate_one_epoch(self,valid_set):
        
        self.model.eval()
        avg_loss = 0.
        
        for val_inputs,val_targets in self.iterator.get_batches(valid_set, shuffle= True):
            #self.optimizer.zero_grad()
            input_vars = torch.from_numpy(val_inputs).float().to(self.device)
            target_vars = torch.from_numpy(val_targets).long().to(self.device)
            target_vars = Variable(target_vars.view(-1))
            out, alphas, spat_attn = self.model(input_vars)
            loss = self.criterion(out, target_vars)
            avg_loss += loss.item()
        return avg_loss/ len(valid_set.X)

          
    def train(self,train_set, valid_set, test_set,batch_size = 32, n_epochs = 400, patience=400):
        print(self.model)

       
        best_acc = 0
        early_stopping = EarlyStopping(verbose=True, patience = 100)
        
        if self.device is not 'cpu':
            self.model = self.model.cuda()
        self.criterion =  nn.CrossEntropyLoss()#nn.MSELoss()#nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad = True,weight_decay=1e-5)
        scheduler = StepLR(self.optimizer, step_size=2, gamma=0.005)
        print("Starting Training of model")
        # Start training loop
        valid_losses = []
        train_losses = []
        for epoch in range(1,n_epochs):

            train_loss, alphas, spat_attn = self.train_one_epoch(train_set)
            valid_loss = self.evaluate_one_epoch(valid_set)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if epoch % 10 == 0:
                print("Epoch{} Avg train Loss: {}, Avg val Loss: {}".format(epoch,train_loss,valid_loss))

            if epoch % 20 == 0:
                test_acc  = evaluate(self.model,test_set, self.iterator)
                if test_acc > best_acc:
                    torch.save(self.model.state_dict(), 'checkpoint.pt')
                    best_acc = test_acc
                print('Epoch {}.......Acc {}'.format(epoch, test_acc))
            """
            if self.early_stopping:
                self.early_stopping(valid_loss, self.model,self.criterion)

            if self.early_stopping.early_stop == True:
                print('Early Stopping epoch{}'.format(epoch))
                acc = evaluate(self.model,valid_set,self.iterator)
                test_acc  = evaluate(self.model,test_set,self.iterator)
                self.model.load_state_dict(torch.load('checkpoint.pt'))
                self.criterion.load_state_dict(torch.load('checkpoint_opt.pt'))
                #scheduler.step()
                self.early_stopping.early_stop = False
        """
        print('done training')
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        #self.criterion.load_state_dict(torch.load('checkpoint_opt.pt'))
        test_acc  = evaluate(self.model,test_set,self.iterator)
        print('Best acc {}'.format(test_acc))
        #print(valid_losses)
        plt.plot(train_losses)
        return test_acc,alphas, spat_attn, self.model
#acc = example()



if __name__== '__main__':
    from analysis import kappa_error,plot_confusion_matrix
    import pandas as pd
    dataset = 'bciiv2b/non_stand'
    acc = list()
    PATH = 'Results'
    models = os.path.join(PATH,'models')
    confusionMatrix = os.path.join(PATH,'confusionMatrix')
    experiments = {}
    
    for i in range(1,10,1):
        subject = os.path.join(models,os.path.split(dataset)[0] +'_'+ str(i)+'.pt')
       
        train_set, test_set = load_data(i, datasetName = os.path.join('../',dataset))
        train_set, test_set = crop(train_set, test_set)
        train_set, valid_set, test_set = prepare_data(train_set, test_set, one_hot= False)
        #train_set, valid_set, test_set = format_for_rnn(train_set, valid_set, test_set)
        print(train_set.X.shape)
        #net = DRNN(n_input = input_dim, n_hidden = 10, n_layers= 3, dropout=0, cell_type='GRU', batch_first=True)
        #net = CONVDRNN()
        net= DemixGRU(classes = 2, inchans = 3)
        
        #net= convrnn(classes = 2, inchans = 3)
        model = Network(model=net,batch_size=32)
        test_acc,alphas, net  = model.train(train_set, valid_set,test_set,n_epochs = 500)
        
        acc.append(test_acc)
        torch.save(net.state_dict(), subject)
         
        _, preds = evaluate(net,test_set,model.iterator, pred = True)
        cm , ov_acc = kappa_error(test_set.y.squeeze(),preds)
        fbeta = cm.F_beta(beta=2)
        fbeta = {k:round(v,2) for k,v in fbeta.items()}
        experiments[i] = [test_acc, cm.Kappa, ov_acc] 
        filename = subject.replace(models,confusionMatrix)
        filename = filename.replace('.pt','.png')
        ax = plot_confusion_matrix(cm)
        ax.figure.savefig(filename)
        del model, train_set, test_set,valid_set
        re = pd.DataFrame(experiments)
        filename = os.path.join(PATH,'1')
        re.to_csv(filename, index= False)