from architecture.modeling.backbone.TemporalStereo import TEMPORALSTEREO
import torch

def flops(Net,imgL,imgR,disp_long_3x,device):
    Net = Net.to(device)
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_long_3x = disp_long_3x.to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net, (imgL, imgR, disp_long_3x))   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params


device = 'cpu'
Net = TEMPORALSTEREO()
imgL = torch.randn([1,3,256,512])
imgR = torch.randn([1,3,256,512])
mem = {'memories': None}

Net = Net.to(device)
imgL = imgL.to(device)
imgR = imgR.to(device)

from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(Net, (imgL, imgR, mem))   # FLOPs（乘加=2）
total_flops = flops.total()

total_params = sum(p.numel() for p in Net.parameters())

print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")