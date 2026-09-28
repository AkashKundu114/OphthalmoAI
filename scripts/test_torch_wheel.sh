#!/bin/bash
set -e
python3 -c "
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

# Robust patch for 1x1 convolutions on Blackwell sm_120 cuBLAS
_orig_conv2d = F.conv2d
def safe_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    is_1x1 = (weight.shape[2:] == (1, 1))
    stride_1 = (stride == 1 or stride == (1, 1))
    pad_0 = (padding == 0 or padding == (0, 0))
    dil_1 = (dilation == 1 or dilation == (1, 1))
    if groups == 1 and is_1x1 and stride_1 and pad_0 and dil_1:
        B, C_in, H, W = input.shape
        C_out = weight.shape[0]
        x_flat = input.permute(0, 2, 3, 1).reshape(-1, C_in)
        w_flat = weight.view(C_out, C_in).t()
        out = x_flat @ w_flat
        if bias is not None:
            out = out + bias
        return out.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    return _orig_conv2d(input, weight, bias, stride, padding, dilation, groups)

F.conv2d = safe_conv2d

import torchvision.models as models

print('--- Testing all 3 backbones with safe 1x1 conv ---')
x = torch.randn(2, 3, 224, 224, device='cuda')

print('1. ConvNeXt Small...')
m1 = models.convnext_small().cuda().eval()
with torch.no_grad():
    out1 = m1(x)
print('  [PASS] ConvNeXt Small shape:', out1.shape)

print('2. DenseNet 201...')
m2 = models.densenet201().cuda().eval()
with torch.no_grad():
    out2 = m2(x)
print('  [PASS] DenseNet 201 shape:', out2.shape)

print('3. EfficientNet V2...')
m3 = models.efficientnet_v2_s().cuda().eval()
with torch.no_grad():
    out3 = m3(x)
print('  [PASS] EfficientNet V2 shape:', out3.shape)

print('\n========================================================================')
print(' SUCCESS! ALL 3 VISION BACKBONES EXECUTED FLAWLESSLY ON RTX 5060!')
print('========================================================================')
"
