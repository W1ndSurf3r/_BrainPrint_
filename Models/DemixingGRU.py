import torch.autograd.function 
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import init

from Models.attn import Attention, Attention2

from Models.ConvGRU import ConvGRU
from Models.SqueezeAndExcitation import SELayer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,7)),
            nn.AvgPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(1,7)),
            nn.AvgPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 120, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        x = x.permute(0, 3, 1, 2)
        xs = self.localization(x)
        #print(xs.size())
        xs = xs.view(-1, 10 * 120)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        #print(theta.size())
        grid = F.affine_grid(theta, x.size())
        
        x = F.grid_sample(x, grid)
        #print('stn', x.size())
        return x
    
    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x
class convLayer(nn.Module):
    
    def __init__(self, cin, cout, kernel_size):
        super(convLayer,self).__init__()
        self.conv = nn.Conv2d(cin,cout,kernel_size, bias = False, padding=(0,kernel_size[1]//2))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        #print(x.size())
        return x
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Demixing(nn.Module):
    
    
    def __init__(self,in_features, out_features, bias=True ):
        
        super(Demixing, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def forward(self, input):
        out = torch.tensordot(input, self.weight,dims= [[1],[0]]) + self.bias
        return out
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)             
                                
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
class DemixGRU(nn.Module):
    
    def __init__(self, classes = 2,inchans = 3):
        super(DemixGRU, self).__init__()
        #self.sn = Net()
        kernels = [(1,7), (1,7)]
        self.conv1 = nn.Conv2d(1,8,(3,9), bias = False)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.ELU()
        #self.demix = Demixing(inchans, 8) # in_features.size(1) the channels
        self.pool =nn.AvgPool2d((1, 10), stride=(1, 10))
        #self.dilgru = DilGRUBlock()
        self.convgru = ConvGRU(8, 8, kernels, 2)
        self.SE = SELayer(3)
        self.attn = Attention2(312,256)
        
         #nn.AdaptiveAvgPool2d((5,20))
         
        self.conv2 = convLayer(32,64,(1,3))
        self.conv3 = convLayer(64,32,(1,3))
        
        self.classifier = nn.Sequential(
            nn.Linear(312, 312),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(312, 2),
            #nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = x.permute(0, 1,4, 2, 3)
        x = x.reshape(x.size(0)*2,x.size(2),x.size(3),x.size(4))
        x = self.SE(x)
        x = self.conv1(x)
        
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        #print('pool',x.size())
        
        #x = self.SE(x)
        x = x.reshape(-1,2, x.size(1),x.size(2),x.size(3))
        
       
        x = self.convgru(x)
        #print(len(x),x[-1].size())
        
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1), x.size(2)*x.size(3)*x.size(4))
        #print('gru',x.size())
        #print(x.size())
        
        #print(x.size())
       
        #print(x.size())
        x , alphas= self.attn(x)
        #print(x.size())
        #x = x.squeeze(-2).permute(0,2,1,3)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.pool(x)
        
      
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.classifier(x)
        #print(x.size())
        return x,0