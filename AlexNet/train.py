import torch as t
import torch.nn as nn

from model import AlexNet
from dutils import train_loader, val_loader

L_RATE = 0.01
MOMENTUM = 0.9
DECAY = 0.0005
EPOCH = 75

device = "cuda" if t.cuda.is_available() else "cpu"
model = AlexNet().to(device)

optim = t.optim.SGD(model.parameters(),lr=L_RATE, momentum=MOMENTUM, weight_decay=DECAY)
lfunc =  nn.CrossEntropyLoss()
lr_scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=15, gamma=0.1)

for epoch in range(45):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = lfunc(model(x), y)
        loss.backward()
        optim.step()
    lr_scheduler.step()

    model.eval()
    correct = total = 0
    with t.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    print(f"epoch {epoch} val acc {correct/total:.3f}")

    
