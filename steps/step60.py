import dezero.dataloaders
from dezero import SeqDataLoader

train_set = dezero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=3)
x, t = next(dataloader)

print(x)
print('----------------')
print(t)