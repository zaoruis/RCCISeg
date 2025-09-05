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

# 图网络卷积
class GraphConvInteraction(nn.Module):
    """
    使用共享权重的图神经网络特征交互模块
    通过权重共享大幅减少参数量，同时保持四个特征的交互能力
    输入: 4个特征张量，每个维度为[b, c, h, w]
    输出: 4个特征张量，每个维度保持[b, c, h, w]不变
    """
    def __init__(self, in_channels):
        super(GraphConvInteraction, self).__init__()
        self.inter_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.self_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.relu = nn.ReLU()
        
    def forward(self, x1, x2, x3, x4):
        # 计算节点间消息（使用共享卷积）
        msg1_to_2 = self.inter_conv(x1)
        msg1_to_3 = self.inter_conv(x1)
        msg1_to_4 = self.inter_conv(x1)
        
        msg2_to_1 = self.inter_conv(x2)
        msg2_to_3 = self.inter_conv(x2)
        msg2_to_4 = self.inter_conv(x2)
        
        msg3_to_1 = self.inter_conv(x3)
        msg3_to_2 = self.inter_conv(x3)
        msg3_to_4 = self.inter_conv(x3)
        
        msg4_to_1 = self.inter_conv(x4)
        msg4_to_2 = self.inter_conv(x4)
        msg4_to_3 = self.inter_conv(x4)
        
        # 聚合消息（使用共享自更新卷积）
        agg1 = self.self_conv(x1) + msg2_to_1 + msg3_to_1 + msg4_to_1
        agg1 = self.relu(agg1)
        
        agg2 = self.self_conv(x2) + msg1_to_2 + msg3_to_2 + msg4_to_2
        agg2 = self.relu(agg2)
        
        agg3 = self.self_conv(x3) + msg1_to_3 + msg2_to_3 + msg4_to_3
        agg3 = self.relu(agg3)
        
        agg4 = self.self_conv(x4) + msg1_to_4 + msg2_to_4 + msg3_to_4
        agg4 = self.relu(agg4)
        
        # 使用共享门控机制融合特征
        out1 = x1 + self.gate_conv(torch.cat([x1, agg1], dim=1))
        out2 = x2 + self.gate_conv(torch.cat([x2, agg2], dim=1))
        out3 = x3 + self.gate_conv(torch.cat([x3, agg3], dim=1))
        out4 = x4 + self.gate_conv(torch.cat([x4, agg4], dim=1))
        
        return out1, out2, out3, out4
    
    
# depth_layer block
class Depth_block(nn.Module):
    def __init__(self, in_channels):
        super(Depth_block, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.GraphConv = GraphConvInteraction(in_channels)
 
    def forward(self, depth):
        n, c, h, w = depth.shape
        input_dwt = self.dwt(depth)
        input_LL, input_HL, input_LH, input_HH = input_dwt[:n, ...], input_dwt[n:2*n, ...], input_dwt[2*n:3*n, ...], input_dwt[3*n:, ...]
        
        output_LL, output_HL, output_LH, output_HH = self.GraphConv(input_LL, input_HL, input_LH, input_HH)
        
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
        indices = self.window_top_low_sample(attn)       # 计算索引
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
    
    
    
    
from thop import profile    

def calculate_model_flops(model, input_shape):
    """
    计算模型的FLOPs和参数量
    
    参数:
        model: PyTorch模型
        input_shape: 输入张量的形状 (batch_size, channels, height, width)
    
    返回:
        flops: 浮点运算次数
        params: 参数量
    """
    # 创建输入张量
    input_tensor = torch.randn(*input_shape)
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    
    return flops, params

if __name__ == "__main__":
    # 初始化模型 - 请替换为你的实际模型
    model = Depth_backbone()
    
    # 输入形状 [batch_size, channels, height, width]
    input_shape = (1, 1, 224, 224)
    
    # 计算模型FLOPs和参数量
    flops, params = calculate_model_flops(model, input_shape)
    
    # 输出结果（可根据需要转换为不同单位）
    print(f"模型FLOPs: {flops:.2f} 次")
    print(f"模型FLOPs (亿次): {flops / 1e8:.2f} 亿次")
    print(f"模型FLOPs (十亿次/GFLOPs): {flops / 1e9:.2f} GFLOPs")
    print(f"模型参数量: {params:.2f} 个")
    print(f"模型参数量 (百万): {params / 1e6:.2f} M")
