from dataset import *
from config import *
from modules import *
from inference import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import torch.optim as optim

def main():
    BATCH_SIZE = 32
    BLOCK_SIZE = 8

    encoder = tiktoken.get_encoding("gpt2")

    xb, yb = get_data(BLOCK_SIZE, BATCH_SIZE, encoder)
    print(f"Data Fetched...")
    config = Config(vocab_size=encoder.n_vocab)
    model = MiniGPT(config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")   

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device {device}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    # checking generation before training
    #print("Checking Inference before training...")
    #sentence = "The sun is shining"
    #res = inference(sentence, model, encoder, device)
    #print(res)
    
    epochs = 500
    steps = 300
    print("Initiating Training...")
    for epoch in range(epochs):
        for i in range(steps):
            model.to(device)
            model.train()
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
        if epoch%50==0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    print("Training Completed...")
    print("Checking Inference after training...")
    sentence = "The sun is shining"
    res = inference(sentence, model, encoder, device)
    print(res)


if __name__ == "__main__":
    main()