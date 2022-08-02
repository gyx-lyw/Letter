import torch
import torch.nn.functional as Fun
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from fwdgrad.loss import xent
import math
import numpy as np
import random

learning_rate = 0.002
k = 0.0003


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


with open("dataset_6_letter.arff", encoding="utf-8") as f:
    header = []
    for line in f:
        if line.startswith("@attribute"):
            header.append(line.split()[1])
        elif line.startswith("@data"):
            break
    df = pd.read_csv(f, header=None)
    df.columns = header
    dataset = df.values
X = dataset[:,:16].astype(float)
Y = dataset[:,16]
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
X = torch.FloatTensor(X)
Y = torch.LongTensor(Y)
train_dataset = torch.utils.data.TensorDataset(X,Y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64)

file = open(f"D:\pythonprogram\experiment\experiment_1.txt", 'a')
class BPNetModel(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(BPNetModel, self).__init__()
        self.hiddden1 = torch.nn.Linear(input_size, hidden1_size)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.out = torch.nn.Linear(hidden2_size, output_size)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        out = self.out(x)
        return out

model = BPNetModel(16, 64, 64, 26)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
model.to(DEVICE)

params = tuple(model.parameters())
it = 0
for epoch in range(200):
     it += 1
     for i, batch in enumerate(train_loader):
         images, labels = batch
         loss = xent(model, images.to(DEVICE), labels.to(DEVICE))
         loss.backward()
         lr = learning_rate * math.e ** (-(epoch * len(train_loader) + i) * k)
         for p in params:
             p.data.sub_(lr * p.grad.data)
             p.grad.data.zero_()
     print(f"第{epoch+1}次训练， Loss: {loss.item():.4f}")
     file.writelines(f'iter:{it},Loss:{loss.item():.4f}\n')
file.close()


