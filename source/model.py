import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataloader
from torch.autograd import Variable



class TransformerWrap(nn.Module):
    def __init__(self):
        super().__init__()
        
