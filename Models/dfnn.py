# DEEP FEEDFORWARD NEURAL NETWORK:
import torch.nn as nn
import torch
import torch.nn.functional as F

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, N_List, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.N_List = N_List
        fc = nn.ModuleList()  # list()
        fc.append(torch.nn.Linear(self.input_size, self.N_List[0]))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(0.2))
        for i in range(0, self.hidden_size - 1):
            fc.append(torch.nn.Linear(self.N_List[i], self.N_List[i + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.2))
        fc.append(torch.nn.Linear(self.N_List[-1], self.output_size))

        self.layers = nn.Sequential(*fc)

    def forward(self, x):
        return F.log_softmax(
                            self.layers(x),
                            dim=1
                            )
