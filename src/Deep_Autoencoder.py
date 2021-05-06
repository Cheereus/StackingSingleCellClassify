import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from tqdm import trange
from sklearn.preprocessing import Normalizer

colors = ['#FF0000', '#FF1493', '#9400D3', '#7B68EE', '#FFD700',
          '#00BFFF', '#00FF00', '#FF8C00', '#FF4500', '#8B4513']

###### 读入数据
digits = load_digits()
x = digits.data
y = digits.target
Y = y  # 在画图中备用

###### 对输入进行归一化，因为autoencoder只用到了input
MMScaler = MinMaxScaler()
x = MMScaler.fit_transform(x)
iforestX = x

###### 输入数据转换成神经网络接受的dataset类型，batch设定为10
tensor_x = torch.from_numpy(x.astype(numpy.float32))
tensor_y = torch.from_numpy(y.astype(numpy.float32))
my_dataset = TensorDataset(tensor_x, tensor_y)
my_dataset_loader = DataLoader(my_dataset, batch_size=10, shuffle=False)


###### 定义一个autoencoder模型
class deep_autoencoder(nn.Module):
    def __init__(self):
        super(deep_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            # torch.randn(10, 64),
            nn.Linear(64, 8),
            # nn.Tanh(),
        )
        self.bottleneck1 = nn.Sequential(
            # torch.randn(10, 8),
            nn.Linear(8, 2),
            # nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 8),
            # nn.ReLU(),
        )
        self.bottleneck2 = nn.Sequential(
            nn.Linear(8, 2),
            # nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        encoder1 = self.encoder1(x)
        bottleneck1 = self.bottleneck1(encoder1)
        decoder1 = self.decoder1(bottleneck1)
        encoder2 = self.encoder2(decoder1)
        bottleneck2 = self.bottleneck2(encoder2)
        decoder2 = self.decoder2(bottleneck2)
        return encoder1, bottleneck1, decoder1, encoder2, bottleneck2, decoder2


model = deep_autoencoder()

####### 定义损失函数
criterion = nn.MSELoss()

####### 定义优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 如果采用SGD的话，收敛不下降

####### epoch 设定为20
for epoch in trange(20):
    total_loss = 0
    for i, (x, y) in enumerate(my_dataset_loader):
        _1, _2, _3, _4, neck2, pred = model(V(x))
        loss = criterion(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    if epoch % 1 == 0:
        print(total_loss.data.numpy())

###### 基于训练好的model做降维并可视化
x_ = []
y_ = []
for i, (x, y) in enumerate(my_dataset):
    _1, _2, _3, _4, neck2, pred = model(V(x))
    # loss = criterion(pred, x)
    dimension = neck2.data.numpy()
    x_.append(dimension[0])
    y_.append(dimension[1])

# 画图
plt.figure(figsize=(10, 10))
plt.xlim(numpy.array(x_).min(), numpy.array(x_).max())
plt.ylim(numpy.array(y_).min(), numpy.array(y_).max())
for i in range(len(numpy.array(x_))):
    plt.text(x_[i], y_[i], str(digits.target[i]), color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('first principle component')
plt.ylabel('second principle component')
plt.show()
