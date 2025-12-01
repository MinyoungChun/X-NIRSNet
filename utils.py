import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, device, criterion=None):

    model.train()
    running_loss = 0.0
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for batch_idx, (x1, x2, y) in enumerate(loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x1, x2)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            outputs = model(x1, x2)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    avg_loss = running_loss / len(loader)
    
    return acc, avg_loss