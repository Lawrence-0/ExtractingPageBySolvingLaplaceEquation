import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd

import torch
import torch.nn as nn
import torch.utils.data as Data

EPOCH = 4000
LR = 0.01
flag = 1
FILE = "test/test.xlsx"

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden_1 = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout()
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden // 2),
            nn.Dropout()
        )
        self.hidden_3 = nn.Sequential(
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden // 4),
            nn.Dropout()
        )
        self.hidden_4 = nn.Sequential(
            nn.Linear(n_hidden // 4, n_hidden // 8),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden // 8),
            nn.Dropout()
        )
        self.out = nn.Linear(n_hidden // 8, n_output)

    def forward(self, pos):
        val = self.hidden_1(pos)
        val = self.hidden_2(val)
        val = self.hidden_3(val)
        val = self.hidden_4(val)
        val = self.out(val)
        return val

def get_data(FILE):
    data = pd.read_excel(FILE)
    data0 = data.loc[0: 877]
    size0 = data.loc[878:879]
    # print(data0)
    # print(size0)
    size0 = pd.DataFrame(size0, columns = ['x', 'y', 'z']).values
    # print(type(size0))
    space = []
    for i in range(size0[0][0], size0[1][0] + 1):
        space.append([])
        for j in range(size0[0][1], size0[1][1] + 1):
            for k in range(size0[0][2], size0[1][2] + 1):
                space[-1].append([i, j, k])
        space[-1] = np.array(space[-1])
        space[-1] = torch.from_numpy(space[-1]).type(torch.FloatTensor)
        # print(space[-1])

    pos = pd.DataFrame(data0, columns = ['x', 'y', 'z'])
    pos = pos.values
    val = pd.DataFrame(data0, columns = ['page'])
    val = val.values
    train_pos = torch.from_numpy(pos).type(torch.FloatTensor)
    label_val = torch.from_numpy(val).type(torch.FloatTensor)
    return train_pos, label_val, space, pos, val



def train():
    train_pos, label_val, space, pos, val = get_data(FILE)
    net = MLP(3, 1024, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.MSELoss()
    if torch.cuda.is_available():
        net = net.cuda()
        train_pos = train_pos.cuda()
        label_val = label_val.cuda()
    cnt = 0
    running_loss = 0
    for epoch in range(EPOCH):
        output = net(train_pos)
        loss = loss_func(output, label_val)                            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()

        if epoch % 4000 == 3999:
            cnt += 1
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
            print(epoch, loss.cpu().data.numpy()), '| run loss: %.4f' % (running_loss / 4000.0))
            print(epoch, output)
            running_loss = 0
            if flag:
                plt.ion()
                plt.cla()
                plt.plot(range(len(pos)), val[:, 0], 'b-', lw=2)
                plt.plot(range(len(pos)), output.cpu().data.numpy()[:, 0], 'r-', lw=1, alpha=0.5)
                plt.text(0.5, 0, 'Loss=%.4f' % loss.cpu().data.numpy(), fontdict={'size': 20, 'color': 'red'})
                plt.show()
    
    torch.save(net,'net.pth')
    print('Finish training')

def test(spc):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train()
    net = torch.load('net1.pth')
    net.to(device)
    spc = spc.to(device)
    rst = net(spc)
    return rst.cpu().detach().numpy()

train_pos, label_val, space, pos, val = get_data(FILE)
volume = []
# train()
for i in range(len(space)):
    volume.append(test(space[i]))
    # print(i)
np.save("volume1.npy",volume)

volume = np.load("volume1.npy")
result = []
for i in range(513):
    result.append([])
    for j in range(513):
        result[-1].append([])
        for k in range(277):
            result[-1][-1].append(volume[i][j * 277 + k])
np.save("result1.npy",result)