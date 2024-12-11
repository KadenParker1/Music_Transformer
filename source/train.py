import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Dataloader
from torch.autograd import Variable
import csv
import os
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

from model import *

# data preprocessor
# encoded = encode_midi('path.mid'), decoded = decode_midi(integerarray, 'safe_path.mid')

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")) # set device, and default to apple chip if available, then gpu, then cpu
print("Running on {}".format(device))

CSV_HEADER = ["Epoch", "Learn Rate", "Avg Train Loss", "Train Accuracy", "Avg Eval Loss", "Eval Accuracy"]



def main(num_epochs, optimizer):
    vocab_size = 5000
    embed_size = 512
    model = TransformerModel(vocab_size, embed_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = LrStepTracker(hidden_dim=embed_size, warm_up_steps=4000)
    num_epochs = 100
    input_dir = " "
    train_dataset, val_dataset, test_dataset = create_datasets(input_dir, )
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=, shuffle=True)
    test_loader = DataLoader()
    val_loader = DataLoader()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(epoch)
        for step, (src, tgt) in enumerate(data_loader):
            src, tgt = src.to(device), tgt.to(device) # setting source and target
            src_mask = None
            tgt_mask = None

            optimizer.zero_grad()
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            lr = lr_scheduler.step(step + epoch * len(data_loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if step % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}, LR: {lr:.6e}")


    

if __name__ == "__main__":
    main()