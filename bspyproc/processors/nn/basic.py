import torch
import torch.nn as nn
import torch.nn.functional as F
from bspyproc.utils.pytorch import TorchUtils

class RingNet(nn.Module):
    def __init__(self, configs):
        super(RingNet, self).__init__()
        self.info = configs
        self.configs = configs
        # linear layer (784 -> 1 hidden node)
        val = 2
        self.fc1 = nn.Linear(2, val)
        self.fc2 = nn.Linear(val, val)
        self.fc3 = nn.Linear(val, 1)
        self.act = nn.Sigmoid()
        if TorchUtils.get_accelerator_type() == torch.device('cuda'):
            self.cuda()
        self.to(TorchUtils.data_type)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        #x = self.act(self.fc3(x))
        x = F.relu(self.fc3(x))

        return x
    
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()