import torch.nn as nn
import torch
import torch.nn.functional
from RektNet.cross_ratio_loss import CrossRatioLoss
from RektNet.resnet import ResNet,ResNet_C_Block,ResNet_D_Block


#-------------------------------ResNet.......................
def print_tensor_stats(x, name):
    flattened_x = x.cpu().detach().numpy().flatten()
    avg = sum(flattened_x)/len(flattened_x)
    print(f"\t\t\t\t{name}: {round(avg,10)},{round(min(flattened_x),10)},{round(max(flattened_x),10)}")

#MIT Implementation uses ResNet
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        #main path 
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        act1 = self.relu1(b1)

        c2 = self.conv2(act1) 
        b2 = self.bn2(c2)

        #shortcut path 
        sc_c = self.shortcut_conv(x)
        sc_b = self.shortcut_bn(sc_c)

        out = self.relu2(b2+sc_b)
        return out

#this is just a chained Input Block to replace the expensive 7x7 filter
class ResNet_C_Block(nn.Module): 
    def __init__(self,in_channels,out_channels): 
            super(ResNet_C_Block,self).__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels//4,kernel_size=3,padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels//4)

            self.conv2 = nn.Conv2d(in_channels=out_channels//4,out_channels=out_channels//2,kernel_size=3,padding=1) 
            self.bn2 = nn.BatchNorm2d(out_channels//2) 

            self.conv3 = nn.Conv2d(in_channels=out_channels//2,out_channels=out_channels,kernel_size=3,padding=1)
            self.bn3 = nn.BatchNorm2d(out_channels) 

            self.ReLu = nn.ReLU()

    def forward(self,x): 

            x = self.ReLu(self.bn1(self.conv1(x)))
            x = self.ReLu(self.bn2(self.conv2(x))) 
            x = self.ReLu(self.bn3(self.conv3(x)))

            return x



#using ResNet-D for Downsampling from the paper "Bag of Tricks for Image Classification"
class ResNet_D_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet_D_Block, self).__init__()

        #resnet path A aka main path 
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu_conv = nn.ReLU()

        #resnet path B aka shortcut path 
        self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1)
        self.pool_shortcut = nn.AvgPool2d(kernel_size=2,stride=2)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        
        self.relu_out = nn.ReLU()

    def forward(self, x):
        
        #main path 
        print(x.shape)
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        a1 = self.relu_conv(b1)

        c2 = self.conv2(a1)
        b2 = self.bn2(c2)
        a2 = self.relu_conv(b2)

        c3 = self.conv3(a2)
        b3 = self.bn3(c3)
        
        print(b3.shape)
        #shortcut 
        sc_p = self.pool_shortcut(x)
        sc_c = self.conv_shortcut(sc_p)
        sc_b = self.bn_shortcut(sc_c) 
        print(sc_b.shape)

        out = self.relu_out(b3+sc_b)
        

#----------------------------------------------------------------------------Neural Net------------------------------------------------------------------------------

class KeypointNet(nn.Module):
    def __init__(self, num_kpt=7, image_size=(80, 80), onnx_mode=False, init_weight=True):
        super(KeypointNet, self).__init__()
        net_size = 16

        #self.conv = nn.Conv2d(in_channels=3, out_channels=net_size, kernel_size=7, stride=1, padding=3)
        #self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.conv = ResNet_C_Block(in_channels=3,out_channels=net_size)

        # torch.nn.init.xavier_uniform(self.conv.weight)
        self.bn = nn.BatchNorm2d(net_size)
        self.relu = nn.ReLU()
        self.res1 = ResNet(net_size, net_size)
        self.res2 = ResNet(net_size, net_size * 2)
        self.res3 = ResNet(net_size * 2, net_size * 4)
        self.res4 = ResNet(net_size * 4, net_size * 8)

        self.out = nn.Conv2d(in_channels=net_size * 8, out_channels=num_kpt, kernel_size=1, stride=1, padding=0)
        # torch.nn.init.xavier_uniform(self.out.weight)
        if init_weight:
            self._initialize_weights()
        self.image_size = image_size
        self.num_kpt = num_kpt
        self.onnx_mode = onnx_mode

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #uses Kaiming/He initalization 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    #Batch Norm init
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                #init of linear layer
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def flat_softmax(self, inp):
        flat = inp.view(-1, self.image_size[0] * self.image_size[1])
        flat = torch.nn.functional.softmax(flat, 1)
        return flat.view(-1, self.num_kpt, self.image_size[0], self.image_size[1])

    def soft_argmax(self, inp):
        values_y = torch.linspace(0, (self.image_size[0] - 1.) / self.image_size[0], self.image_size[0], dtype=inp.dtype, device=inp.device)
        values_x = torch.linspace(0, (self.image_size[1] - 1.) / self.image_size[1], self.image_size[1], dtype=inp.dtype, device=inp.device)
        exp_y = (inp.sum(3) * values_y).sum(-1)
        exp_x = (inp.sum(2) * values_x).sum(-1)
        return torch.stack([exp_x, exp_y], -1)

    def forward(self, x):
        act1 = self.relu(self.bn(self.conv(x)))
        
        act2 = self.res1(act1)
        act3 = self.res2(act2)
        act4 = self.res3(act3)    
        act5 = self.res4(act4)
        hm = self.out(act5)
        #add one res block
        #act5 = self.res5(act5)

        if self.onnx_mode:
            return hm
        else:
            hm = self.flat_softmax(self.out(act5))
            out = self.soft_argmax(hm)
            return hm, out.view(-1, self.num_kpt, 2)


class KeypointNetD(nn.Module):
    def __init__(self, num_kpt=7, image_size=(80, 80), onnx_mode=False, init_weight=True):
        super(KeypointNetD, self).__init__()
        net_size = 16

        self.conv = nn.Conv2d(in_channels=3, out_channels=net_size, kernel_size=7, stride=1, padding=3)
        # torch.nn.init.xavier_uniform(self.conv.weight)
        self.bn = nn.BatchNorm2d(net_size)
        self.relu = nn.ReLU()
        self.res1 = ResNet(net_size, net_size)
        self.res2 = ResNet(net_size, net_size * 2)
        self.res3 = ResNet(net_size * 2, net_size * 4)
        self.res4 = ResNet(net_size * 4, net_size * 8)
        self.out = nn.Conv2d(in_channels=net_size * 8, out_channels=num_kpt, kernel_size=1, stride=1, padding=0)
        # torch.nn.init.xavier_uniform(self.out.weight)
        if init_weight:
            self._initialize_weights()
        self.image_size = image_size
        self.num_kpt = num_kpt
        self.onnx_mode = onnx_mode

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #uses Kaiming/He initalization 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    #Batch Norm init
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                #init of linear layer
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def flat_softmax(self, inp):
        flat = inp.view(-1, self.image_size[0] * self.image_size[1])
        flat = torch.nn.functional.softmax(flat, 1)
        return flat.view(-1, self.num_kpt, self.image_size[0], self.image_size[1])

    def soft_argmax(self, inp):
        values_y = torch.linspace(0, (self.image_size[0] - 1.) / self.image_size[0], self.image_size[0], dtype=inp.dtype, device=inp.device)
        values_x = torch.linspace(0, (self.image_size[1] - 1.) / self.image_size[1], self.image_size[1], dtype=inp.dtype, device=inp.device)
        exp_y = (inp.sum(3) * values_y).sum(-1)
        exp_x = (inp.sum(2) * values_x).sum(-1)
        return torch.stack([exp_x, exp_y], -1)

    def forward(self, x):
        act1 = self.relu(self.bn(self.conv(x)))
        act2 = self.res1(act1)
        act3 = self.res2(act2)
        act4 = self.res3(act3)
        act5 = self.res4(act4)
        hm = self.out(act5)
        if self.onnx_mode:
            return hm
        else:
            hm = self.flat_softmax(self.out(act5))
            out = self.soft_argmax(hm)
            return hm,out.view(-1, self.num_kpt, 2)

if  __name__=='__main__':
    from torch.autograd import Variable
    from torch import autograd
    net = KeypointNet()
    test = net(Variable(torch.randn(3, 3, 80, 80)))
    loss = CrossRatioLoss()
    target = autograd.Variable(torch.randn(3, 7, 2))
    l = loss(test, target)
