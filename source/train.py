import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataloader
from torch.autograd import Variable

# data preprocessor
from process_midi import * # encoded = encode_midi('path.mid'), decoded = decode_midi(integerarray, 'safe_path.mid')

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")) # set device, and default to apple chip if available, then gpu, then cpu
print("Running on {}".format(device))


# set up model


# set up dataloader

def get_train_data_batch():

    pass


def train():
