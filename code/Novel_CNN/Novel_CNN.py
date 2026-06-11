import torch
import torch.nn as nn
import torch.nn.functional as F


class Novel_CNN(nn.Module):
    def __init__(self, datanum):
        """
        datanum: 信号长度 (EOG通常为512, EMG通常为1024)
        """
        super(Novel_CNN, self).__init__()

        # 定义基础卷积块 (Conv-Conv-Pool)
        def conv_block(in_f, out_f, pool=True):
            layers = [
                nn.Conv1d(in_f, out_f, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(out_f, out_f, kernel_size=3, padding=1),
                nn.ReLU()
            ]
            if pool:
                layers.append(nn.AvgPool1d(kernel_size=2))
            return nn.Sequential(*layers)

        # 1. Block 1: 32 channels, pool
        self.block1 = conv_block(1, 32)  # Out: datanum/2

        # 2. Block 2: 64 channels, pool
        self.block2 = conv_block(32, 64)  # Out: datanum/4

        # 3. Block 3: 128 channels, pool
        self.block3 = conv_block(64, 128)  # Out: datanum/8

        # 4. Block 4: 256 channels, dropout, pool
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(2)  # Out: datanum/16
        )

        # 5. Block 5: 512 channels, dropout, pool
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(2)  # Out: datanum/32
        )

        # 6. Block 6: 1024 channels, dropout, pool
        self.block6 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(2)  # Out: datanum/64
        )

        # 7. Block 7: 2048 channels, dropout (No Pool)
        self.block7 = nn.Sequential(
            nn.Conv1d(1024, 2048, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 8. 全连接层
        # 根据池化层数量（6个），最终长度为 datanum // 64
        flatten_dim = 2048 * (datanum // 64)
        self.fc = nn.Linear(flatten_dim, datanum)

        # 初始化权重 (对应 TF 的 he_normal)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (Batch, 1, datanum)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.unsqueeze(1)  # 回到 (Batch, 1, datanum)