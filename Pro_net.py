import torch

from torch import nn

# batch = 4
# C_in = 4
img_size = 100**2
sub_size = 50**2
exten = 32
lcom = 2

class ResNet(nn.Module):
    def __init__(self,input_size, hidden_size, block, output_size, batch):
        super(ResNet, self).__init__()
        self.batch = batch
        self.inputsize = input_size

        self.down = nn.Sequential(
            nn.Conv2d(input_size, exten, stride=1, kernel_size=5, padding=2),
            nn.BatchNorm2d(exten),
            nn.PReLU(),
            # nn.Conv2d(exten, exten//4, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(exten//4),
            # nn.Conv2d(exten, exten, stride=1, kernel_size=5, padding=2),
            # nn.BatchNorm2d(exten),
            # nn.Conv2d(exten, exten//4, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(exten//4),
            # nn.Conv2d(16, 8, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8),
            # nn.Conv2d(8, 4, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4),
            # nn.Conv2d(exten//4, exten//2, stride=1, kernel_size=5, padding=2),
            # nn.BatchNorm2d(exten//2),
            nn.Conv2d(exten, input_size, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size),
            nn.Tanh()
            # nn.Conv2d(exten // 2, 1, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1),
        )
        #
        self.MLP = nn.Sequential(
            nn.Conv1d(img_size,img_size,kernel_size=1),
            nn.BatchNorm1d(img_size),
            nn.PReLU(),
            # nn.Conv1d(sub_size, sub_size//lcom,kernel_size=1),
            # nn.BatchNorm1d(sub_size//lcom),
            nn.Conv1d(img_size, img_size//4, kernel_size=1),
            nn.BatchNorm1d(img_size//4),
            nn.Tanh(),
            # nn.Conv1d(sub_size, 4*sub_size,kernel_size=1),
            # nn.LazyBatchNorm1d(),
            nn.Conv1d(img_size//4, sub_size, kernel_size=1),
            nn.BatchNorm1d(sub_size),
            nn.Tanh()

        )

        # 第一层卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.PReLU()
        )
        num_of_layers = block

        # 中间残差层
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(ResidualBlock(hidden_size))
        self.residuals = nn.Sequential(*layers)

        # 输出层
        self.out_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size)
        )

        self.out_layer_2 = nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1)
        # self.out_layer_2 = nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1)


    def forward(self, x):
        fx = self.down(x)
        dx=fx.view(self.batch,img_size,self.inputsize)
        lin = self.MLP(dx)
        # lin2 = self.MLP(lin1) + lin1
        # lin  = self.MLP(lin2) + lin2
        out = lin.permute(0,2,1)
        out=torch.reshape(out,(self.batch,self.inputsize,50,50))
        block1 = self.block1(out)
        # block1 = self.block1(fx)
        # res_out1 = self.residuals(block1) + block1
        # res_out2 = self.residuals(res_out1) + res_out1
        # res_out3 =  self.residuals(res_out2) + res_out2
        # res_out = self.residuals(res_out2) + res_out2
        res_out = self.residuals(block1)
        out_layer_1 = self.out_layer(res_out) #+ res_out
        # out_layer_1 = self.out_layer(block1)  # + res_out
        block8 = self.out_layer_2(block1 + out_layer_1) #% (2*torch.pi)
        # nn.Dropout(0.5)
        # block8 = torch.sin(self.out_layer_2(block1 + out_layer_1))
        # block8 = block6%(2*torch.pi)
        # result1 = block8
        # result2 = (block8+self.dp)%(torch.pi/2)
        # results = torch.concat((result1, result2),axis=1)
        return block8
        # return results
#
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.prelu = nn.PReLU()
        self.Tanlu = nn.Tanh()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        # residual = self.prelu(residual)
        residual = self.Tanlu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
