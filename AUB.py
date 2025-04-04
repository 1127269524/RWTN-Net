import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_blocks import CBAM
from BMVC import MultiScaleGatedAttn
def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src



class Conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True) if relu else None
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x







class PConv2d_(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 dilation=1,
                 n_div: int = 2):
        super(PConv2d_, self).__init__()
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        self.conv = nn.Conv2d(in_channels=self.dim_conv,
                              out_channels=self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=dilation,
                              bias=False)#填充等于膨胀导致尺寸不变
        self.CBAM = CBAM(in_channels)


    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.CBAM(x)

        return x

#逐点卷积加BN和激活函数
class PWConv(nn.Module):
    def __init__(self, in_planes, out_planes, act=None,kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(PWConv, self).__init__()
        self.act_ = act
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        if act == None:
            self.bn  = nn.Identity()
            self.act = nn.Identity()  #代表该层不做操作
        else:
            self.bn = nn.BatchNorm2d(out_planes)
            if act == 'gelu':
                self.act = nn.GELU()
            elif act == 'relu':
                self.act = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        if self.act_ != None:
            x = self.bn(x)
            x = self.act(x)
        return x


# 输出和输入通道数一致 进行pconv和两次逐点卷积，第一次逐点有激活函数
class Dilate_FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dilation : int,
                 act: str = 'gelu',
                 mid = 2
                 ):
        super(Dilate_FasterNetBlock, self).__init__()

        self.conv1 = PConv2d_(in_channels,dilation=dilation)#不影响尺寸
        self.conv2 = PWConv(in_channels,in_channels*mid,act=act)
        self.conv3 = PWConv(in_channels*mid,in_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + y





class Faster_RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12):
        super(Faster_RSU4F,self).__init__()

        self.rebnconvin = PWConv(in_ch,mid_ch,act='gelu',kernel_size=3,padding=1)#输入通道数从 in_ch 转变为 mid_ch，同时保持尺寸不变，经过激活函数处理。

        self.rebnconv1 = Dilate_FasterNetBlock(mid_ch,dilation=1)#进行pconv和两次逐点卷积，第一次逐点有激活函数
        self.rebnconv2 = Dilate_FasterNetBlock(mid_ch,dilation=2)#有激励函数
        self.rebnconv3 = Dilate_FasterNetBlock(mid_ch,dilation=4)#有激励函数

        self.rebnconv4 = Dilate_FasterNetBlock(mid_ch,dilation=8)##有激励函数，都不影响尺寸[4,mid_ch=12,320,320]

        self.rebnconv3d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=4,padding=4)#改变通道数 2mid->mid
        self.rebnconv2d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=2,padding=2)#改变通道数 2mid->mid
        self.rebnconv1d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)##有激励函数，都不影响尺寸[4,mid_ch=12,320,320]

    def forward(self,x):
        hx = x

        hxin = self.rebnconvin(hx)#改变通道不影响尺寸大小，经过激励函数
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))#将两个特征图按照通道拼接。然后经过膨胀卷积，并且带有激励函数。
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin






class Faster_RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Faster_RSU6,self).__init__()

        self.rebnconvin = PWConv(in_ch,mid_ch,act='gelu',kernel_size=3,padding=1)#改变通道不影响尺寸大小

        self.rebnconv1 = Dilate_FasterNetBlock(mid_ch,dilation=1)#进行pconv和两次逐点卷积，第一次逐点有激活函数，通道数不变
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)#长宽变为原来一半，ceil_mode代表不够除时候向上取整

        self.rebnconv2 = Dilate_FasterNetBlock(mid_ch,dilation=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = Dilate_FasterNetBlock(mid_ch,dilation=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = Dilate_FasterNetBlock(mid_ch,dilation=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = Dilate_FasterNetBlock(mid_ch,dilation=1)

        self.rebnconv6 = Dilate_FasterNetBlock(mid_ch,dilation=2)#输出和输入通道数一致，尺寸也不变

        self.rebnconv5d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv4d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv3d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv2d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv1d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)

        self.MultiScaleGatedAttn1=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn2=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn3=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn4=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn5=MultiScaleGatedAttn(mid_ch)



    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)#通道变为mid

        hx1 = self.rebnconv1(hxin)#通道不变为mid 这是原来尺寸
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(self.MultiScaleGatedAttn5(hx6,hx5))#先融合再将通道减半最终是mid
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(self.MultiScaleGatedAttn4(hx5dup,hx4))#先融合再将通道减半最终是mid
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(self.MultiScaleGatedAttn3(hx4dup,hx3))#先融合再将通道减半最终是mid
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(self.MultiScaleGatedAttn2(hx3dup,hx2))#先融合再将通道减半最终是mid
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(self.MultiScaleGatedAttn1(hx2dup,hx1))##先融合再将通道减半最终是mid
        return hx1d + hxin

    def forward_2(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d, hx2d, hx3d, hx4d, hx5d, hx6

