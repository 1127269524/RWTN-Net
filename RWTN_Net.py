import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from AUB import Faster_RSU4F,Faster_RSU6
from FRFE import FRFE
from SConv import SConv_2d
from ASFF import  ASFF2,ASFF3
from LASCC import LASCC


class GFFM(nn.Module):
    def __init__(self,c1=64,c2=128,c3=256,c4=512):
        super(GFFM, self).__init__()

        self.linear_c4 = nn.Conv2d(c4, 32, 1)
        self.linear_c3 = nn.Conv2d(c3, 32, 1)
        self.linear_c2 = nn.Conv2d(c2, 32, 1)
        self.linear_c1 = nn.Conv2d(c1, 32, 1)

        #AUB4
        self.up_linear_fuse = Faster_RSU4F(128, 64)

        self.gated1 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                    torch.nn.AdaptiveMaxPool2d((1,1), return_indices=False),
                                    torch.nn.Sigmoid()
                                    )
        self.gated2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                    torch.nn.AdaptiveAvgPool2d((1,1)),
                                    torch.nn.Sigmoid()
                                    )


        self.outconv = nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.down8 = nn.AvgPool2d(kernel_size=8, stride=8)


    def forward(self, c1, c2, c3, c4):#160,80,40,20


        _c4 = self.linear_c4(c4)
        up_c4 = self.up8(_c4)
        down_c4 = _c4


        _c3 = self.linear_c3(c3)
        up_c3 = self.up4(_c3)
        down_c3 = self.down2(_c3)


        _c2 = self.linear_c2(c2)
        up_c2 = self.up2(_c2)
        down_c2 = self.down4(_c2)

        _c1 = self.linear_c1(c1)
        up_c1 = _c1
        down_c1 = self.down8(_c1)

        x = torch.cat([up_c4, up_c3, up_c2, up_c1], dim=1)   # 上采样

        gate1 = self.gated1(torch.cat([down_c4, down_c3, down_c2, down_c1], dim=1)) # 下采样后门控



        x = x + x * gate1

        x = self.outconv(self.up_linear_fuse(x))


        return x


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val



class atrous_spatial_pyramid_pooling_GELU(nn.Module):
    def __init__(self, in_channel, depth=128, rate_dict=[2, 4, 8, 12]):
        super(atrous_spatial_pyramid_pooling_GELU, self).__init__()

        self.modules = []
        for index, n_rate in enumerate(rate_dict):
            self.modules.append(nn.Sequential(
                nn.Conv2d(in_channel, depth, 3, dilation=n_rate, padding=n_rate, bias=False),
                nn.BatchNorm2d(depth),
                nn.GELU(),
                nn.Conv2d(depth, int(depth / 4), 1, dilation=n_rate, bias=False),
            ))

        self.convs = nn.ModuleList(self.modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[1]=res[0]+res[1]
        res[2]=res[1]+res[2]
        res[3]=res[2]+res[3]
        res[4]=res[3]+res[4]
        res[5]=res[4]+res[5]
        return torch.cat(res, 1)

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src






class RWTN_Net(nn.Module):
    def __init__(self,Coefficient_3):
        super(RWTN_Net, self).__init__()
        self.Coefficient_3=Coefficient_3
        self.backbone = FRFE(Coefficient_3=Coefficient_3)
        #返回[b,40,160,160]
        self.ASFF2_40_80=ASFF2(level=1, multiplier=1, rfb=False, vis=False, dim=[80,40])
        #返回[b,160,20,20]
        self.ASFF2_160_160=ASFF2(level=0, multiplier=1, rfb=False, vis=False, dim=[160,160])
        #返回[b,80,80,80]
        self.ASFF3_40_80_160=ASFF3(level=1, multiplier=1, rfb=False, vis=False, dim=[160,80,40])
        #返回[b,160,40,40]
        self.ASFF3_80_160_160=ASFF3(level=1, multiplier=1, rfb=False, vis=False, dim=[160,160,80])

        self.head = GFFM(c1=40,c2=80,c3=160,c4=160)
        self.corr1 = LASCC(topk=32)  # 原 my_mobile_Corr(topk=32)


        self.side1 = nn.Conv2d(32, 1, 3, padding=1)
        self.side2 = SConv_2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,same=True)
        self.side3 = SConv_2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,same=True)


        self.aspp = nn.Sequential(nn.LayerNorm([32,80,80]),
            atrous_spatial_pyramid_pooling_GELU(in_channel=32, depth=64, rate_dict=[2, 4, 8, 12,16,20]),
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.rsu = Faster_RSU6(96,64)

        self.outconv = nn.Conv2d(4, 1, 1)


    def forward(self, x):
        x0 = x
        #FRFE
        x1, x2, x3, x4 = self.backbone.forward_feature(x)
        #AMAF
        x11=x1
        x22=x2
        x33=x3
        x44=x4
        x1=self.ASFF2_40_80([x11,x22])
        x2=self.ASFF3_40_80_160([x11,x22,x33])
        x3=self.ASFF3_80_160_160([x22,x33,x44])
        x4=self.ASFF2_160_160([x33,x44])
        d0 = self.head(x1, x2, x3, x4)   # (b,128,80,80)

        #LASCC
        d0 = self.corr1(d0)    # (b,32,80,80)
        d1=d0
        #PASPP
        s1 = self.aspp(d0)

        y=torch.cat([d1,s1],dim=1)#96

        #AUB6
        s2 = self.rsu(y)

        d2 = _upsample_like(self.side2(s1,self.Coefficient_3), x0)

        d3 = _upsample_like(self.side3(s2,self.Coefficient_3), x0)


        return d3,d2


