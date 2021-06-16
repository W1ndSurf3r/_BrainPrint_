
import torch.autograd.function 
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import init

class Attention(nn.Module):
    
    def __init__(self,hidden_size, attn_size):
        
        super(Attention, self).__init__()
        
        self.attn_size = attn_size
        self.w_omega = Parameter(torch.Tensor(hidden_size,attn_size))
        self.b_omega = Parameter(torch.Tensor(attn_size))
        self.u_omega = Parameter(torch.Tensor(attn_size))
        self.reset_parameters()
        
    def forward(self, input_):
        #print('input',input_.size(), 'weight', self.w_omega.size())
        out = torch.tensordot(input_, self.w_omega,dims= 1) + self.b_omega
        #print('output', out.size())
        v = torch.tanh(out)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        #print('vu', vu.size())
        alphas = torch.softmax(vu, dim=1)
        test = input_ * alphas.unsqueeze(-1)
        #print('Alphas', alphas.size(), input_.size(),test.size())
        output = torch.sum(input_ * alphas.unsqueeze(-1), dim=1)
        #print('final output', output.size())
        return output, alphas
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w_omega, std=0.1)
       
        torch.nn.init.normal_(self.b_omega, std=0.1)      
        torch.nn.init.normal_(self.u_omega, std=0.1) 
                                
    def extra_repr(self):
        return 'Attention size Weights={}, bias={}'.format(
            self.w_omega.size(),  self.b_omega.size())
    
class Attention2(nn.Module):
    
    def __init__(self,hidden_size, attn_size):
        
        super(Attention2, self).__init__()
        
        self.attn_size = attn_size
        self.w_omega = Parameter(torch.Tensor(hidden_size,attn_size))
        self.b_omega = Parameter(torch.Tensor(attn_size))
        self.u_omega = Parameter(torch.Tensor(attn_size))
        self.reset_parameters()
        
    def forward(self, input_):
        
        #print('input',input_.size(), 'weight', self.w_omega.size())
        out = torch.tensordot(input_, self.w_omega,dims= 1) + self.b_omega
        #print('output', out.size())
        v = torch.tanh(out)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        #print('vu', vu.size())
        alphas = torch.softmax(vu, dim=1)
        test = input_ * alphas.unsqueeze(-1)
        #print('Alphas', alphas.size(), input_.size(),test.size())
        output = torch.sum(input_ * alphas.unsqueeze(-1), dim=1)
        #print('final output', output.size())
        return output, alphas
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w_omega, std=0.1)
       
        torch.nn.init.normal_(self.b_omega, std=0.1)      
        torch.nn.init.normal_(self.u_omega, std=0.1) 
                                
    def extra_repr(self):
        return 'Attention size Weights={}, bias={}'.format(
            self.w_omega.size(),  self.b_omega.size())