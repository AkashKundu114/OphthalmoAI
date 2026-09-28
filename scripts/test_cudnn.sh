python3 -c "
import torch
import torch.nn as nn
print('--- Testing with torch.backends.cudnn.enabled = False ---')
torch.backends.cudnn.enabled = False

conv = nn.Conv2d(3, 64, 3).cuda()
inp = torch.randn(4, 3, 64, 64, device='cuda', requires_grad=True)
out = conv(inp)
print('Forward OK! Output shape:', out.shape)
loss = out.sum()
loss.backward()
print('Backward OK! Weight grad shape:', conv.weight.grad.shape)
print('Input grad shape:', inp.grad.shape)
print('ALL CUDA FORWARD & BACKWARD TESTS PASSED ON RTX 5060 LAPTOP GPU!')
"
