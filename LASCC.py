import torch
import torch.nn as nn
import torch.nn.functional as F

def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class ZeroWindow: # 给输入加权，削弱特征图中每个像素点自身及附近范围对自身的相关性。根据高斯分布削弱
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1): #x_in是相似度矩阵,被reshape成了(b, w*h, h, w),h和w是相似度矩阵前面的特征图的高宽
        sigma = h * rat_s, w * rat_s
        b, c, h2, w2 = x_in.shape
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
            ind_r = torch.arange(h2, device=x_in.device).view(1, 1, h2, 1)
            ind_c = torch.arange(w2, device=x_in.device).view(1, 1, 1, w2)
            # ind_r和ind_c可以联合表示格子的位置

            c_ind_r, c_ind_c = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=x_in.device),
                torch.arange(w, dtype=torch.float32, device=x_in.device),
                indexing="ij"
            )
            cent_r = c_ind_r.reshape(1, c, 1, 1).to(x_in.device)
            cent_c = c_ind_c.reshape(1, c, 1, 1).to(x_in.device)

            def fn_gauss(x, u, s):  # 高斯分布函数
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g
        else:
            out_g = self.store[key]
        out = out_g * x_in
        return out

class LASCC(nn.Module):
    def __init__(self, topk=3,patch_size=2):
        super().__init__()

        self.topk = topk
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 2))
        self.conv_transpose = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
        self.zero_window = ZeroWindow()

        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def forward(self, x):  # x (b,c,h,w)
        patch_size = self.patch_size
        b,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        num_patch_h = int(h/patch_size)
        num_patch_w = int(w/patch_size)
        num_patches = num_patch_w*num_patch_h
        patch_area = patch_size*patch_size   # 一个patch包含几个像素点
        assert num_patches == (h*w)/patch_area


        x = F.normalize(x, p=2, dim=-3)  # 在通道维度归一化


        # 下面的N指patch个数num_patches，P指一个patch包含几个像素点patch_area
        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(b * c * num_patch_h, patch_size, num_patch_w, patch_size)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(b, c, num_patches, patch_area)# 这一步是(8，128，1600，4)

        i = torch.arange(patch_size)
        diagonal_indices = i * (patch_size + 1)  # 对角线元素的线性索引
        x = x[:, :, :, diagonal_indices]  # [B, C, N, P], K=对角线元素数量


        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [B*P, N, C]
        x = x.reshape(b * patch_size, num_patches, -1)

        x = torch.matmul(x, x.transpose(1, 2))
        x = x.reshape(b,patch_size, num_patches, num_patches)
        channel_avg = x.mean(dim=1, keepdim=True)  # 结果形状为 [B, 1, N, N]


        x=torch.cat([x,channel_avg],dim=1)


        # zero out same area corr
        x = self.zero_window(x.view(b*(patch_size+1), -1, num_patch_h, num_patch_w), num_patch_h, num_patch_w, rat_s=0.05).reshape(b,(patch_size+1), num_patch_h*num_patch_w, num_patch_h*num_patch_w)

        x = F.softmax(x * self.alpha, dim=-1) * F.softmax(x * self.alpha, dim=-2)   #归一化 （BP,N,N）
        new_x = x[:,-1,:,:].unsqueeze(1).expand(b, patch_area, num_patches, num_patches).clone()
        for index, diag_index in enumerate(diagonal_indices):
            new_x[:,diag_index , :, :] = x[:,index, :, :]
        x=new_x


        # [B, P, N, N] -> [B, N, N, P]
        x = x.transpose(1, 3)
        # [B, N, N, P] -> [B*N*n_h, n_w, p_h, p_w]
        x = x.reshape(b * num_patches * num_patch_h, num_patch_w, patch_size, patch_size)
        # [B*N*n_h, n_w, p_h, p_w] -> [B*N*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*N*n_h, p_h, n_w, p_w] -> [B, N, H, W]
        x = x.reshape(b, num_patches, num_patch_h * patch_size, num_patch_w * patch_size)

        x = get_topk(x, k=self.topk, dim=-3)  #在h1*w1维度选取top_k


        return x