import torch
import torch.nn as nn

# 1. 全连接网络
class fcNN(nn.Module):
    def __init__(self, datanum):
        super(fcNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x.unsqueeze(1)

# 2. LSTM 网络
class RNN_lstm(nn.Module):
    def __init__(self, datanum):
        super(RNN_lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum)
        )

    def forward(self, x):
        x = x.transpose(1, 2) # (B, 512, 1)
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out.unsqueeze(1)

# 3. Simple CNN
class Simple_CNN(nn.Module):
    def __init__(self, datanum):
        super(Simple_CNN, self).__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        self.features = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64)
        )
        self.fc = nn.Linear(64 * datanum, datanum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(1)

# 4. Complex CNN (ResNet 结构)
class Res_BasicBlock(nn.Module):
    def __init__(self, kernelsize):
        super(Res_BasicBlock, self).__init__()
        pad = kernelsize // 2
        self.bblock = nn.Sequential(
            nn.Conv1d(32, 32, kernelsize, padding=pad),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernelsize, padding=pad),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernelsize, padding=pad),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
    def forward(self, x):
        return x + self.bblock(x)

class BasicBlockall(nn.Module):
    def __init__(self):
        super(BasicBlockall, self).__init__()
        self.b3 = nn.Sequential(Res_BasicBlock(3), Res_BasicBlock(3))
        self.b5 = nn.Sequential(Res_BasicBlock(5), Res_BasicBlock(5))
        self.b7 = nn.Sequential(Res_BasicBlock(7), Res_BasicBlock(7))
    def forward(self, x):
        return torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)

class Complex_CNN(nn.Module):
    def __init__(self, datanum):
        super(Complex_CNN, self).__init__()
        self.pre = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.body = BasicBlockall()
        self.post = nn.Sequential(nn.Conv1d(96, 32, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.fc = nn.Linear(32 * datanum, datanum)
    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = self.post(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(1)