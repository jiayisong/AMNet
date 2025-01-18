import torch
import torch.nn.functional as F
'''
from mmcv.ops.modulated_deform_conv_deterministic import ModulatedDeformConv2dDeterministicPack


torch.set_printoptions(threshold=torch.inf)
a = ModulatedDeformConv2dDeterministicPack(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False).cuda()
#a.weight.data.fill_(-1)
b = torch.rand([1,1,4,4]).cuda()
print(a.weight)
c = a(b)
c.sum().backward()
print(b)
print(c)
'''
'''
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
offset = (torch.rand([2, 2, 10]) * 2).cuda()
input = torch.rand([2, 2, 2, 2]).cuda()
B, C, h, w = input.shape
t1 = torch.floor(offset)
t2 = t1[:, 0, :] + t1[:, 1, :] * w
t2_sort, t2_indice = torch.sort(t2, 1)
t2_change = t2_sort[:, 1:] - t2_sort[:, :-1]
t2_mask = torch.cat((t2_change.new_ones([B, 1], dtype=torch.bool), t2_change > 0), 1)
batch_id, t2_i_i_s_id = torch.where(t2_mask)
t2_i_i_f_id = torch.cat((t2_i_i_s_id[1:], t2_i_i_s_id.new_zeros([1, ])), 0)
loc_number = t2_sort[t2_mask]
w_number = loc_number % w
h_number = torch.div(loc_number, w, rounding_mode="trunc")
w_ji = (w_number % 2).bool()
w_ou = ~w_ji
h_ji = (h_number % 2).bool()
h_ou = ~h_ji
g1 = w_ou * h_ou
batch_id1, t2_i_i_s_id1, t2_i_i_f_id1 = batch_id[g1], t2_i_i_s_id[g1], t2_i_i_f_id[g1]
print(offset)
print(batch_id1.dtype)
print(t2_i_i_s_id1.dtype)
print(t2_i_i_f_id1.dtype)
print(t2_indice.dtype)
'''
'''
from mmcv.ops.grid_sample_deterministic import grid_sample_deterministic, ModulatedDeformConv2dFastv2Pack
from mmcv.ops.modulated_deform_conv_deterministic import ModulatedDeformConv2dDeterministicPack
import torch.nn as nn

input1 = torch.rand([2,200,50,50], requires_grad=True).cuda()
input2 = torch.rand([2,200,50,50], requires_grad=True).cuda()
input2.data = input1.data[:]
input1.retain_grad()
input2.retain_grad()
a = ModulatedDeformConv2dFastv2Pack(200, 200, 1, 1, 0, bias=False).cuda()

b = ModulatedDeformConv2dDeterministicPack(200, 200, 1, 1, 0, bias=False).cuda()
a.conv.weight.data = b.weight.data[:]
nn.init.uniform_(b.conv_offset.weight, -0.01, 0.01)
a.conv_offset.weight.data = b.conv_offset.weight.data[:]
c = a(input1)
d = b(input2)
(c.sum()+d.sum()).backward()

print(input1.grad.abs().mean())
print(c.abs().mean())
print((input1.grad-input2.grad).abs().max())
print((c-d).abs().max())
'''
'''
gr = []
for i in range(2):
    SEED = 2
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 为所有GPU设置随机种子
    from mmcv.ops.grid_sample_deterministic import grid_sample_deterministic, ModulatedDeformConv2dFastv2Pack

    torch.set_printoptions(threshold=torch.inf)
    input1 = torch.rand([2, 1, 10, 10], requires_grad=True).cuda()
    input1.retain_grad()
    offset1 = torch.rand([2, 2, 1000], requires_grad=True).cuda() * 12 - 1
    offset1.retain_grad()
    a = grid_sample_deterministic(input1, offset1)
    a.sum().backward()
    gr.append(input1.grad)
'''
'''
from mmcv.ops.grid_sample_deterministic import grid_sample_deterministic

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
B,C,H,W = 8, 128, 40, 40
input1 = torch.rand([B,C,H,W], requires_grad=True).cuda()
input1.retain_grad()
offset1 = torch.rand([B, 2, H*W], requires_grad=True).cuda() * (H + 2) - 1
offset1.retain_grad()
output1 = torch.nn.functional.grid_sample(input1, offset1.transpose(1, 2).reshape(B,H,W,2)*2/(W-1) -1, align_corners=True).view(B,C,H*W)
output1.sum().backward()

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
B,C,H,W = 8, 128, 40, 40
input2 = torch.rand([B,C,H,W], requires_grad=True).cuda()
input2.retain_grad()
offset2 = torch.rand([B, 2, H*W], requires_grad=True).cuda() * (H + 2) - 1
offset2.retain_grad()
output2 = grid_sample_deterministic(input2, offset2)
output2.sum().backward()
print(input1)
print(offset1)
print(output1)
print(output2)
print((output1-output2).abs().max())
print((input1.grad-input2.grad).abs().max())
print((offset1.grad-offset2.grad).abs().mean())
'''
'''
from mmcv.ops.grid_sample_deterministic import grid_sample_deterministic, DeformUpSample

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
B,C,H,W = 1, 1, 4, 4
input1 = torch.rand([B,C,H,W], requires_grad=True).cuda()
input1.retain_grad()
D =  DeformUpSample(C, ).cuda()
output = D(input1)
print(input1)
print(output)
D.conv_offset.bias.data.fill_(0.5)
output = D(input1)
print(output)
'''
'''
a = torch.rand([1,8,2,2])
b = F.pixel_shuffle(a, 2)
print(a)
print(b)
'''
from mmcv.ops.grid_sample_deterministic import grid_sample_deterministic, DeformTransConv

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
B,C,H,W = 1, 1, 4, 4
COUT=1
input1 = torch.rand([B,C,H,W], requires_grad=True).cuda()
input1.retain_grad()
D =  DeformTransConv(C, COUT, 3, 4, 1, mask=False).cuda()
print(input1)
a = torch.zeros_like(D.conv.weight)
a = a.view(COUT, C, 3,3)
a[:,:,1,1] = 1
a = a.view(D.conv.weight.shape)
D.conv.weight.data = a
output = D(input1)
print(output)
#output = D(input1)
#print(output)
