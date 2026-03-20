from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as TF

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=0,hidden_depth=0,act=nn.ReLU):
        super().__init__()
        if hidden_depth == 0:
            mods = [nn.Linear(input_dim, output_dim)]
        else:
            if act is not None:
                mods = [nn.Linear(input_dim,hidden_dim),act()]
                for i in range(hidden_depth - 1):
                    mods += [nn.Linear(hidden_dim, hidden_dim),act()]
            else:
                mods = [nn.Linear(input_dim,hidden_dim)]
                for i in range(hidden_depth - 1):
                    mods += [nn.Linear(hidden_dim, hidden_dim)]
            mods.append(nn.Linear(hidden_dim,output_dim))
        self.net = nn.Sequential(*mods)

    def zero_last_layer(self):
        with torch.no_grad():
            *_,last_module = self.net
            last_module.weight.data = torch.zeros_like(last_module.weight.data)

    def forward(self,tensor_input):
        return self.net(tensor_input)

class FunctionalMLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=0,hidden_depth=0,act=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self,tensor_input,model_params):
        weight = model_params[:len(model_params)//2]
        bias = model_params[len(model_params)//2:]
        net_result = TF.linear(tensor_input,weight[0],bias[0])
        for layer_idx in range(self.hidden_depth):
            net_result = TF.linear(net_result,weight[layer_idx+1],bias[layer_idx+1])
            if (layer_idx < self.hidden_depth-1) and (self.act is not None):
                net_result = self.act.forward(net_result)
        return net_result

    def export_init_params(self):
        weight_list = []
        bias_list = []
        input_size_sqrt = 2*sqrt(self.input_dim)
        hidden_size_sqrt = 2*sqrt(self.hidden_dim)
        if self.hidden_depth == 0:
            weight_list.append(torch.rand((self.output_dim,self.input_dim)))
            bias_list.append(torch.rand(self.output_dim)*input_size_sqrt-input_size_sqrt/2)
        else:
            weight_list.append(torch.rand((self.hidden_dim,self.input_dim))*input_size_sqrt-input_size_sqrt/2)
            bias_list.append(torch.rand(self.hidden_dim)*input_size_sqrt-input_size_sqrt/2)
            for layer_idx in range(self.hidden_depth-1):
                weight_list.append(torch.rand((self.hidden_dim,self.hidden_dim))*hidden_size_sqrt-hidden_size_sqrt/2)
                bias_list.append(torch.rand(self.hidden_dim)*hidden_size_sqrt-hidden_size_sqrt/2)
            weight_list.append(torch.rand((self.output_dim,self.hidden_dim)))
            bias_list.append(torch.rand(self.output_dim)*hidden_size_sqrt-hidden_size_sqrt/2)
        return weight_list+bias_list