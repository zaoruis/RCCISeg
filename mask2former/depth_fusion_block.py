import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np


# 数据预处理Item
class Item(nn.Module):
    def __init__(self):
        super(Item, self).__init__()
        self.conv1 = models.resnet50(pretrained=True).conv1
        self.bn1 = models.resnet50(pretrained=True).bn1
        self.relu = models.resnet50(pretrained=True).relu
        self.maxpool = models.resnet50(pretrained=True).maxpool
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

# 二维离散小波变换
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)
    
# 二维离散小波逆变换
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    
    return h

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

# depth_layer block
class Depth_block(nn.Module):
    def __init__(self, in_channel):
        super(Depth_block, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.conv1 = nn.Conv2d(in_channels = in_channel, 
                               out_channels = in_channel, 
                               kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels = in_channel, 
                               out_channels = in_channel, 
                               kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels = in_channel, 
                               out_channels = in_channel, 
                               kernel_size=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels = in_channel, 
                               out_channels = in_channel, 
                               kernel_size=1, stride=1, bias=True)
 
    def forward(self, depth):
        n, c, h, w = depth.shape
        input_dwt = self.dwt(depth)
        input_LL, input_HL, input_LH, input_HH = input_dwt[:n, ...], input_dwt[n:2*n, ...], input_dwt[2*n:3*n, ...], input_dwt[3*n:, ...]
        
        output_LL = self.conv1(input_LL)
        output_HL = self.conv2(input_HL)
        output_LH = self.conv3(input_LH)
        output_HH = self.conv4(input_HH)
        
        output_depth = torch.cat((output_LL, output_HL, output_LH, output_HH), dim = 0)
        depth = self.iwt(output_depth)
        return depth

# depth backbone
class Depth_backbone(nn.Module):
    def __init__(self):
        super(Depth_backbone, self).__init__()
        # resnet作为backbone
        in_channels = [256, 512, 1024, 2048]  
        # swin-s作为backbone
        #in_channels = [96, 192, 384, 768]            
        self.item = Item()
        self.block0 = Depth_block(in_channels[0])
        self.block1 = Depth_block(in_channels[1])
        self.block2 = Depth_block(in_channels[2])
        self.block3 = Depth_block(in_channels[3])
        
        self.conv0 = nn.Conv2d(in_channels = 64, 
                               out_channels = in_channels[0], 
                               kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels = in_channels[0], 
                               out_channels = in_channels[1], 
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels = in_channels[1], 
                               out_channels = in_channels[2], 
                               kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels = in_channels[2], 
                               out_channels = in_channels[3], 
                               kernel_size=1, stride=1)
        self.maxpool = models.resnet50(pretrained=True).maxpool
    
    def forward(self, x):
        x = torch.cat((x, x, x), dim = 1)
        x = self.item(x)

        x0 = self.block0(self.conv0(x))
        x1 = self.maxpool(self.block1(self.conv1(x0)))
        x2 = self.maxpool(self.block2(self.conv2(x1)))
        x3 = self.maxpool(self.block3(self.conv3(x2)))
        
        return [x0, x1, x2, x3]




# 基于不同的采样方法
class Sampler(nn.Module):
    def __init__(self, in_channels, size, ratio = 0.5):
        super(Sampler, self).__init__()
        self.size = size
        self.ratio = ratio
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size = 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False)
        )
    
    # 随机采样
    def random_sample(self, attn_map):
        B, H, W = attn_map.shape
        total_pixels = H * W 

        num_samples = int(total_pixels * self.ratio)
        all_samples = []

        for b in range(B):
            random_indices = torch.randperm(total_pixels)[:num_samples]

            h_coords = random_indices // W 
            w_coords = random_indices % W
            coords = torch.stack([h_coords, w_coords], dim=1)
            
            all_samples.append(coords)
            
        return all_samples

    # 等距采样
    def equidistant_sample(self, attn_map):
        B, H, W = attn_map.shape
        total_pixels = H * W
        num_samples = int(total_pixels * self.ratio)

        # 计算 stride
        approx_stride = (H * W / num_samples) ** 0.5
        stride = max(1, int(approx_stride))

        all_samples = []

        for b in range(B):
            coords = []
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    coords.append([i, j])
            # 若超过采样点数，随机截取（或直接取前 N 个）
            if len(coords) > num_samples:
                indices = torch.randperm(len(coords))[:num_samples]
                coords = [coords[idx] for idx in indices]
            else:
                coords = coords[:num_samples]  # 不足时也不会报错
            coords = torch.tensor(coords, dtype=torch.long, device=attn_map.device)
            all_samples.append(coords)
            
        return all_samples

    # 全局top-low采样
    def global_top_low_sample(self, attn_map):
        """
        全局最大最小值采样（不分块），带熵自适应调整比例。
        :param attn_map: Tensor, shape [B, H, W]
        :return: List of [N_i, 2] tensors，表示每个样本的采样点位置（行列坐标）
        """
        B, H, W = attn_map.shape
        indices_all = []

        for b in range(B):
            flat = attn_map[b].flatten()  # [H*W]
            num_pixels = flat.numel()
            k = int(num_pixels * self.ratio)

            # 计算归一化熵
            prob = F.softmax(flat, dim=0)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
            if num_pixels == 1:
                norm_entropy = 0.1
            else:
                norm_entropy = max(0.1, entropy / torch.log(torch.tensor(float(num_pixels))).item())

            # Top-K 和 Low-K 的个数
            k_top = int(round(k * norm_entropy))
            k_low = k - k_top

            # 最大值和最小值采样
            top_idx = torch.topk(flat, k_top, largest=True)[1]
            low_idx = torch.topk(flat, k_low, largest=False)[1]

            selected_idx = torch.cat([top_idx, low_idx], dim=0)  # shape: [k]

            # 转换为坐标（H, W）
            global_h = selected_idx // W
            global_w = selected_idx % W
            coords = torch.stack([global_h, global_w], dim=1)  # [k, 2]

            indices_all.append(coords)

        return indices_all  # List of [k, 2]

    # 基于窗口的top-low采样
    def window_top_low_sample(self, attn_map):
        B, H, W = attn_map.shape
        indices_all = []

        for b in range(B):
            indices_b = []

            for i in range(0, H, self.size):
                for j in range(0, W, self.size):
                    h_end = min(i + self.size, H)
                    w_end = min(j + self.size, W)

                    block = attn_map[b, i:h_end, j:w_end]  # [h, w]
                    num_pixels = block.numel()
                    k = int(num_pixels * self.ratio)

                    prob = F.softmax(block.flatten(), dim=0)  # 展平后softmax
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
                    if num_pixels == 1:
                        norm_entropy = 0.1
                    else :
                        norm_entropy = max(0.1, entropy / torch.log(torch.tensor(float(num_pixels))).item())  # 最大熵为 log(N)
                
                    k_top = int(round(k * norm_entropy))
                    k_low = k - k_top

                    flat = block.flatten()

                    top_vals, top_idx = torch.topk(flat, k_top, largest=True)
                    low_vals, low_idx = torch.topk(flat, k_low, largest=False)

                    selected_idx = torch.cat([top_idx, low_idx], dim=0)

                    local_h = selected_idx // (w_end - j)
                    local_w = selected_idx % (w_end - j)
                    global_h = local_h + i
                    global_w = local_w + j

                    coords = torch.stack([global_h, global_w], dim=1)
                    indices_b.append(coords)

            indices_all.append(torch.cat(indices_b, dim=0))

        return indices_all  # List of [N_i, 2]

    # 0填充
    def scatter_sparse_features(self, b, indices):
        B, C, H, W = b.shape
        device = b.device
        sparse = torch.zeros(B, C, H, W, device=device, dtype=b.dtype)

        for b_idx in range(B):
            coords = indices[b_idx]  # [N, 2]
            h_idx = coords[:, 0]
            w_idx = coords[:, 1]
            sparse[b_idx, :, h_idx, w_idx] = b[b_idx, :, h_idx, w_idx]

        return sparse

    # 前向传播
    def forward(self, a, b):
        attn = self.attn_conv(a).squeeze(1)  # [B, H, W]
        indices = self.equidistant_sample(attn)       # 计算索引
        sparse_b = self.scatter_sparse_features(b, indices)  # 构建稀疏图
        weight = self.fc(self.avg_pool(sparse_b))
        return weight

# 3WC
class WWWC(nn.Module):
    def __init__(self, in_channels, size):
        super(WWWC, self).__init__()
        # 交替拼接后通道数会翻倍，因此卷积层输入通道数为2*in_channels
        self.conv1 = nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        
        self.weight = Sampler(in_channels=in_channels, size=size)

    # 经纬度交错连接   
    def interleave_channels(self, a, b):
        """将两个特征图沿通道维度交替拼接 [a1,b1,a2,b2,...]"""
        # 确保输入特征图尺寸匹配
        assert a.shape == b.shape, "输入特征图a和b的形状必须完全一致"
        batch_size, channels, height, width = a.shape

        result = torch.zeros(
            (batch_size, 2 * channels, height, width),
            device=a.device,
            dtype=a.dtype
        )

        result[:, ::2] = a
        result[:, 1::2] = b 
        
        return result
    
    # 前向传播
    def forward(self, a, b):
        interleaved = self.interleave_channels(a, b)
        x = self.conv2(self.conv1(interleaved))
        weight = self.weight(a, b)
        output = weight * x + a + b
        return output




class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.conv1 = models.resnet50(pretrained=True).conv1
        self.bn1 = models.resnet50(pretrained=True).bn1
        self.relu = models.resnet50(pretrained=True).relu
        self.maxpool = models.resnet50(pretrained=True).maxpool
        self.layer1 = models.resnet50(pretrained=True).layer1
        self.layer2 = models.resnet50(pretrained=True).layer2
        self.layer3 = models.resnet50(pretrained=True).layer3
        self.layer4 = models.resnet50(pretrained=True).layer4
 
    def forward(self, x):
        x = torch.cat((x, x, x), dim = 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]
