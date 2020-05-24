import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        self.bn1_1 = nn.BatchNorm2d(8)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        self.bn2_2 = nn.BatchNorm2d(16)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(24)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.bn3_2 = nn.BatchNorm2d(24)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(40)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(80)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        #class score branch
        self.conv4_2_cls = nn.Conv2d(40, 80, 3, 1, 1)
        self.bn4_2_cls = nn.BatchNorm2d(80)
        self.ip1_cls = nn.Linear(4 * 4 * 80, 128)
        self.ip2_cls = nn.Linear(128, 128)
        self.ip3_cls = nn.Linear(128, 2)
        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.prelu4_2_cls = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.preluip1_cls = nn.PReLU()
        self.preluip2_cls = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        #input shape:[N,3,113,113]
        x = self.ave_pool(self.prelu1_1(self.bn1_1(self.conv1_1(x)))) #[N,8,27,27]
        # block 2
        x = self.prelu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.prelu2_2(self.bn2_2(self.conv2_2(x))) #[N,16,25,25]
        x = self.ave_pool(x) #[N,16,12,12]
        # block 3
        x = self.prelu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.prelu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.ave_pool(x)
        # block 4
        x = self.prelu4_1(self.bn4_1(self.conv4_1(x)))
        # points branch
        ip3_landmark = self.prelu4_2(self.bn4_2(self.conv4_2(x)))
        ip3_landmark = ip3_landmark.view(-1, 4 * 4 * 80)
        ip3_landmark = self.preluip1(self.ip1(ip3_landmark))
        ip3_landmark = self.preluip2(self.ip2(ip3_landmark))
        ip3_landmark = self.ip3(ip3_landmark)
        #face score branch
        ip3_cls = self.prelu4_2_cls(self.bn4_2_cls(self.conv4_2_cls(x)))
        ip3_cls = ip3_cls.view(-1, 4 * 4 * 80)
        ip3_cls = self.preluip1_cls(self.ip1(ip3_cls))
        ip3_cls = self.preluip2_cls(self.ip2(ip3_cls))
        ip3_cls = self.ip3_cls(ip3_cls)
        return ip3_landmark,ip3_cls

if __name__=="__main__":
    model =  BaseNet()
    from thop import profile
    from thop import clever_format
    import torch
    input = torch.randn(1, 1, 112, 112)
    ip3,ip3_cls = model(input)
    print(ip3.shape,ip3_cls.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("param:", params, "flops:", flops)