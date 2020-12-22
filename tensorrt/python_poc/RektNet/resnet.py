import torch.nn as nn

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
        


