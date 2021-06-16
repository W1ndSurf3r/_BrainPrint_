import torch.autograd.function 
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import init
from Models.attn import Attention
from Models.SqueezeAndExcitation import SELayer
from Models.midBlock import midBlock


class convLayer(nn.Module):
    
    def __init__(self, cin, cout, kernel_size, dense = False):
        super(convLayer,self).__init__()
        self.dense = dense
        self.conv = nn.Conv2d(cin,cout,kernel_size, bias = True, padding = (0, kernel_size[1]//2))
        self.bn = nn.BatchNorm2d(cout,momentum=0.993, eps=1e-5)
        self.act = nn.LeakyReLU(True)
        
    def forward(self,x):
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dense:
            #print(x.size(), identity.size())
            x = torch.cat((x,identity),dim=1)
        #print(x.size())
        return x


# Can i add multi-head pooling? and if so, how useful is that.
# Run with within-subject and across-subjects


        
class convrnn(nn.Module):
    
    def __init__(self, classes = 2, inchans = 3):
        super(convrnn, self).__init__()
        kernel_1st	= (1,25)
        #self.norm = LayerNorm(400)
        #self.sn = Net()
        
        self.conv1 = convLayer(1,1,kernel_1st) # Shape is now (B,C, 1,461)
        self.SE = SELayer(inchans, reduction = 8)
        self.conv1_1 = convLayer(1,40,(inchans,3))
        self.pool0 = nn.MaxPool2d((1,3),stride=(1,3))
        
        self.conv2 = midBlock(in_channels=40, output=60, kernel_size =(1,3),pool_size=3)#convLayer(40,40,(1,3), dense=False)
        self.conv3 = midBlock(in_channels=60, output=40, kernel_size =(1,3),pool_size=3)
        #self.conv4 = midBlock(in_channels=80, output=40, kernel_size =3,pool_size=3)
        
        self.drop0 = nn.Dropout(p=0.5)
        
        
        self.rnn = nn.GRU(240, 64, 2,bidirectional= False, batch_first = True, dropout= 0.5) #400: 520. 200: 240
        
        self.attn = Attention(64,256)
        self.drop1 = nn.Dropout(p=0.5)
        self.Linear = nn.Linear(64,classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        #x =self.norm(x)
        x = x.permute(0, 1,4, 2, 3)

        x = x.reshape(x.size(0)*6,x.size(2),x.size(3),x.size(4))

        spat_attn = 0
        x = self.conv1(x)
        #x, spat_attn = self.SE(x)
        x = self.conv1_1(x)
        #x = torch.cat((x1,x2),dim=1)
        x = self.pool0(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.reshape(-1,6, x.size(1)*x.size(2)*x.size(3))
        
        #print('back',x.shape)
        output, states = self.rnn(x) # seq_len, batch, num_directions * hidden_size
        attention, alphas = self.attn(output)
        #print(alphas)
        attention = self.drop1(attention)
    
        output = self.Linear(attention)
        
        #print(output.size())
        return output, alphas, spat_attn