import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable

# 假设这些模块在你的项目中位置如下（根据原代码推断）
from model.utils import resnet
from model.kpff import FeatureFuser, SensoryUpdater, MultiscaleSensoryUpdater

class ImageEncoder(nn.Module):
    def __init__(self, encoder_type: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=pretrained)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [Batch, Channel, Height, Width]
        f2 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f4 = self.layer1(f2)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        return f16, f8, f4


class MaskEncoder(nn.Module):
    def __init__(
        self,
        pix_dim: int,
        value_dim: int,
        sensory_dim: int,
        encoder_type: str = 'resnet18',
        pretrained: bool = True,
    ):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=pretrained, extra_dim=2)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=pretrained, extra_dim=2)
        else:
            raise NotImplementedError

        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        embed_dim = {'resnet18': 256, 'resnet50': 1024}[encoder_type]
        self.fuser = FeatureFuser(pix_dim, embed_dim, value_dim)
        self.sensory_updater = SensoryUpdater(value_dim, sensory_dim)

        # 重新定义 conv1 以适应输入维度变化（如适用）
        # 注意：原代码最后有一行 self.conv1 = ... 这看起来是覆盖了 resnet 的 conv1
        # 如果这是意图保留的行为，则保留如下：
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(
        self,
        image_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        masks_BO2HW: torch.Tensor,
        sensory_BOCHW: torch.Tensor,
        *,
        deep_update: bool = False,
        chunk_size=-1
    ) -> torch.Tensor:
 
        B, N, K, H, W = masks_BO2HW.shape
        assert K == 2, f"Expected K=2 for masks, but got K={K}"

        image_BO1HW = image_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        input_BO3HW = torch.cat([image_BO1HW, masks_BO2HW], dim=2)

        if chunk_size < 1 or chunk_size > N:
            chunk_size = N

        if deep_update and chunk_size != N:
            new_sensory_BOCHW = torch.empty_like(sensory_BOCHW)
        else:
            new_sensory_BOCHW = sensory_BOCHW.clone()

        X_chunks = []
        for i in range(0, N, chunk_size):
            X_BM3HW = input_BO3HW[:, i:i+chunk_size] 
            X_D5HW = X_BM3HW.flatten(start_dim=0, end_dim=1) 
            
            # Forward pass through CNN layers
            X_DCHW = self.maxpool(self.relu(self.bn1(self.conv1(X_D5HW))))
            X_DCHW = self.layer1(X_DCHW) 
            X_DCHW = self.layer2(X_DCHW) 
            X_DCHW = self.layer3(X_DCHW) 
            
            X_BMCHW = X_DCHW.view(B, X_BM3HW.shape[1], *X_DCHW.shape[1:])
            X_BMCHW = self.fuser(pixfeat_BCHW, X_BMCHW)
            X_chunks.append(X_BMCHW)

            if deep_update:
                sensory_chunk_BMCHW = sensory_BOCHW[:, i:i+chunk_size]
                new_sensory_BOCHW[:, i:i+chunk_size] = \
                    self.sensory_updater(X_BMCHW, sensory_chunk_BMCHW)

        X_BNCHW = torch.cat(X_chunks, dim=1)
        return X_BNCHW, new_sensory_BOCHW


class MaskUpsampleBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        if in_dim == out_dim:
            self.linear_proj = nn.Identity()
        else:
            self.linear_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.scale_factor = scale_factor

    def upsample(
            self, feat: torch.Tensor, ratio: float,
            mode: str = 'bilinear', align_corners: bool = False
    ) -> torch.Tensor:
        B, N = feat.shape[:2]
        feat = F.interpolate(
            feat.flatten(start_dim=0, end_dim=1),
            scale_factor=ratio, mode=mode, align_corners=align_corners)
        feat = feat.view(B, N, *feat.shape[1:])
        return feat

    def forward(self, feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        B, N = feat.shape[:2]
        feat_BNCHW, skip_BCHW = feat, skip_feat
        feat_BNCHW = self.upsample(feat_BNCHW, ratio=self.scale_factor)
        skip_BNCHW = skip_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        feat_BNCHW = feat_BNCHW + skip_BNCHW
        feat_DCHW = feat_BNCHW.flatten(start_dim=0, end_dim=1)
        feat_DCHW = self.linear_proj(feat_DCHW) + \
                    self.act2(self.conv2(self.act1(self.conv1(feat_DCHW))))
        feat_BNCHW = feat_DCHW.view(B, N, *feat_DCHW.shape[1:])
        return feat_BNCHW


class MaskDecoder(nn.Module):
    def __init__(self, ms_dims, up_dims, sensory_dim):
        super().__init__()
        self.linear_proj_f8 = nn.Conv2d(ms_dims[1], up_dims[0], kernel_size=1)
        self.linear_proj_f4 = nn.Conv2d(ms_dims[2], up_dims[1], kernel_size=1)
        
        # 使用本文件定义的 Block
        self.upsample_16_to_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.upsample_8_to_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])
        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=1)

        self.sensory_updater = MultiscaleSensoryUpdater(
            up_dims, sensory_dim, sensory_dim)

    def forward(
        self, 
        ms_feats: Iterable[torch.Tensor],
        readout_BNCHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        update_sensory: bool=False,
        chunk_size: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = readout_BNCHW.shape[:2]
        v16_BNCHW = readout_BNCHW
        f8_BCHW, f4_BCHW = ms_feats[1:] 
        f8_BCHW = self.linear_proj_f8(f8_BCHW)
        f4_BCHW = self.linear_proj_f4(f4_BCHW)

        chunk_size = N if chunk_size < 1 or chunk_size > N else chunk_size
        
        if update_sensory:
            if chunk_size != N:
                new_sensory_BNCHW = torch.empty_like(sensory_BNCHW)
            else:
                new_sensory_BNCHW = sensory_BNCHW.clone()
        else:
            new_sensory_BNCHW = sensory_BNCHW

        logits_chunks = []
        for i in range(0, N, chunk_size):
            v16_chunk_BMCHW = v16_BNCHW[:, i:i+chunk_size]
            M = v16_chunk_BMCHW.shape[1] 
            v8_chunk_BMCHW = self.upsample_16_to_8(v16_chunk_BMCHW, f8_BCHW)
            v4_chunk_BMCHW = self.upsample_8_to_4(v8_chunk_BMCHW, f4_BCHW)
            v4_chunk_DCHW = v4_chunk_BMCHW.flatten(start_dim=0, end_dim=1)
            logits_chunk_D1HW = self.pred(v4_chunk_DCHW) 
            logits_chunk_BMHW = \
                logits_chunk_D1HW.view(B, M, *logits_chunk_D1HW.shape[-2:])

            if update_sensory:
                logits_chunk_BM1HW = logits_chunk_BMHW.unsqueeze(2)
                v4_chunk_BMCHW = torch.cat([v4_chunk_BMCHW, logits_chunk_BM1HW], dim=2)
                new_sensory_BNCHW[:, i:i+chunk_size] = self.sensory_updater(
                    [v16_chunk_BMCHW, v8_chunk_BMCHW, v4_chunk_BMCHW], 
                    sensory_BNCHW[:, i:i+chunk_size])

            logits_chunks.append(logits_chunk_BMHW)
        logits_BNCHW = torch.cat(logits_chunks, dim=1)

        return logits_BNCHW, new_sensory_BNCHW