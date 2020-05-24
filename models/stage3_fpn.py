import torch.nn as nn
import torch.nn.functional as F
import torch

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # nn.ReLU(inplace=True),#知乎上说此处去掉relu会涨点

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class MobileNetV1FPN(nn.Module):
    def __init__(self):
        super(MobileNetV1FPN, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(1, 8, 2),
            conv_dw(8, 16, 1),
            conv_dw(16, 16, 2),
            conv_dw(16, 24, 1),
            conv_dw(24, 24, 2),
            conv_dw(24, 40, 1)
        )
        self.stage2 = nn.Sequential(
            conv_dw(40, 40, 2),
            conv_dw(40, 56, 1),
            conv_dw(56, 56, 1),
            conv_dw(56, 56, 1),
            conv_dw(56, 56, 1),
            conv_dw(56, 56, 1)
        )
        self.stage3 = nn.Sequential(
            conv_dw(56, 56, 2),
            conv_dw(56, 96, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.merge16 = conv_bn(152, 128)
        self.merge8 = conv_bn(168, 128)
        self.cls = nn.Linear(96, 2)
        self.landmark = nn.Linear(352, 42)

    def forward(self, x):
        #bottom to top
        x_8 = self.stage1(x) #stride 8
        x_16 = self.stage2(x_8) #stride 16
        x_32 = self.stage3(x_16) #stride 32
        x_32_cls = self.avg(x_32)
        x_32_cls = x_32_cls.view(-1,96)
        cls = self.cls(x_32_cls)
        #top tp bottom
        up_16 = F.interpolate(x_32, size=[x_16.size(2), x_16.size(3)], mode="nearest")
        new_16 = torch.cat((x_16,up_16),dim=1)
        new_16 = self.merge16(new_16)
        up_8 = F.interpolate(new_16, size=[x_8.size(2), x_8.size(3)], mode="nearest")
        new_8 = torch.cat((x_8,up_8),dim=1)
        new_8 = self.merge8(new_8)
        x_8 = self.avg(new_8)
        x_16 = self.avg(new_16)
        x_32 = self.avg(x_32)
        x=torch.cat((x_8,x_16,x_32),dim=1)
        x = x.view(-1, 352)
        landmark = self.landmark(x)
        return landmark,cls

if __name__=="__main__":
    model =  MobileNetV1FPN()
    from thop import profile
    from thop import clever_format
    import torch
    input = torch.randn(8, 1, 112, 112)
    landmark,cls = model(input)
    print(landmark.shape,cls.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("param:", params, "flops:", flops)