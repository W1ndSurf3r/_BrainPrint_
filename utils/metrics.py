from sklearn.metrics import classification_report
from pycm import *
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def kappa_error(y_pred,y_test) -> np.float64:
   
    cm = ConfusionMatrix(y_test, y_pred, digit=2)

    return cm, cm.Overall_ACC

def count_parameters(model):
    for p in model.parameters():
        print(p.shape)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(cm,normalize=True,title='Confusion matrix',annot=True,cmap="YlGnBu"):
    plt.figure()
    if normalize == True:
        df = pd.DataFrame(cm.normalized_matrix).T.fillna(0)
    else:
        df = pd.DataFrame(cm.matrix).T.fillna(0)
    ax = sns.heatmap(df,annot=annot,cmap=cmap)
    ax.set_title(title)
    ax.set(xlabel='Predict', ylabel='Actual')
    return ax
def predict_classes(model, test, raw= True):
    #print(test.X.shape)
    input_vars = np_to_var(test.X , pin_memory=False)
    if raw == True:
        input_vars= input_vars.unsqueeze(-1)
    input_vars = input_vars.cuda()
    #print(input_vars.size())
    out= model(input_vars)
    out = var_to_np(out)
    #print("out:",out.shape)
    #print(np.argmax(out))
    pred_labels = [np.argmax(o, axis=0) for o in out]
    return pred_labels