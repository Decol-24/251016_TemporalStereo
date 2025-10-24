import torch
import argparse
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time

def evaluate_time(Net,imgL,imgR,device,**kwargs):
    import time

    Net = Net.to(device)
    imgL = imgL.to(device)
    imgR = imgR.to(device)

    for i in range(10):
        preds = Net(imgL, imgR)

    times = 30
    start = time.perf_counter()
    for i in range(times):
        preds = Net(imgL, imgR)
    end = time.perf_counter()

    avg_run_time = (end - start) / times

    return avg_run_time
    

def step_time(args,Net,train_loader,val_loader,**kwargs):
    assert args.batch_size == 1

    Net = Net.to(args.device)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
        imgL, imgR = imgL.to(args.device), imgR.to(args.device)
        break

    for i in range(10):
        preds = Net(imgL, imgR)

    Net.t.reset()

    for i in range(30):
        preds = Net(imgL, imgR)

    print(Net.t.all_avg_time_str(30))

def disable_history(net):
    if hasattr(net, 'with_previous'):
        net.with_previous = False
    if hasattr(net, 'frame_idxs'):
        try: net.frame_idxs = set()
        except: pass

def flops(model, device='cpu', bs=1, c=3, H=256, W=512, t=0):
    from fvcore.nn import FlopCountAnalysis

    model = model.to('cpu').eval()   # 计 FLOPs 用 CPU 最省心
    disable_history(model)

    batch = make_dummy_batch(bs, c, H, W, t, device='cpu')
    outputs = {}

    # fvcore 会多次调用 forward，所以 outputs 每次都新建更安全
    class Wrapper(torch.nn.Module):
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self): 
            b = make_dummy_batch(bs, c, H, W, t, device='cpu')
            return self.net.forward(b, {}, is_train=False, timestamp=t)

    wrapper = Wrapper(model)
    f = FlopCountAnalysis(wrapper,())  # 我们把 forward 设计成无参
    total_flops = f.total()
    return total_flops

def make_dummy_batch(bs=1, c=3, H=256, W=512, t=0, device='cuda'):
    batch = {}
    batch[('color_aug', t, 'l')] = torch.randn(bs, c, H, W, device=device)
    batch[('color_aug', t, 'r')] = torch.randn(bs, c, H, W, device=device)
    return batch

class NetWrapper(nn.Module):
    def __init__(self, net, bs=1, c=3, H=256, W=512, t=0, device='cpu'):
        super().__init__()
        self.net = net
        self.bs, self.c, self.H, self.W, self.t = int(bs), int(c), int(H), int(W), int(t)
        self.device = torch.device(device)

    def forward(self):
        # 每次 forward 都新建 batch/outputs，避免 fvcore 多次调用产生副作用
        batch = make_dummy_batch(self.bs, self.c, self.H, self.W, self.t, self.device)
        outputs = {}
        return self.net.forward(batch, outputs, is_train=False, timestamp=self.t)
    
@torch.no_grad()
def benchmark_once(model, H=256, W=512, bs=1, c=3, t=0, warmup=10, iters=50, amp=False):
    device = next(model.parameters()).device
    model.eval()

    # 避免走到 update_map 分支（省掉前后帧依赖）
    if hasattr(model, 'with_previous'):
        model.with_previous = False     # 关键：别走 update_map 分支
    if hasattr(model, 'frame_idxs'):
        # 确保 (timestamp-1) 不在 frame_idxs 里
        try:
            model.frame_idxs = set()    # 或者 []、tuple()
        except Exception:
            pass

    # 预热
    for _ in range(warmup):
        batch = make_dummy_batch(bs, c, H, W, t, device)
        outputs = {}
        if amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                _ = model.forward(batch, outputs, is_train=False, timestamp=t)
        else:
            _ = model.forward(batch, outputs, is_train=False, timestamp=t)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 正式计时（CUDA 用 event 更准）
    times_ms = []
    use_events = device.type == 'cuda'
    if use_events:
        starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)

    for _ in range(iters):
        batch = make_dummy_batch(bs, c, H, W, t, device)
        outputs = {}
        if use_events:
            starter.record()
        t0 = time.time()
        if amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                _ = model.forward(batch, outputs, is_train=False, timestamp=t)
        else:
            _ = model.forward(batch, outputs, is_train=False, timestamp=t)
        if use_events:
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))  # 纯GPU时间
        else:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times_ms.append((time.time() - t0) * 1000.0)

    import numpy as np
    print(f'[{device.type}] N={iters} | mean={np.mean(times_ms):.3f} ms | std={np.std(times_ms):.3f} ms '
          f'| bs={bs}, shape=({c},{H},{W}), amp={amp}')
    return times_ms

@torch.no_grad()
def run_once(model, H=256, W=512, bs=1, c=3, t=0, warmup=10, iters=50, amp=False):
    device = next(model.parameters()).device
    model.eval()

    # 避免走到 update_map 分支（省掉前后帧依赖）
    if hasattr(model, 'with_previous'):
        model.with_previous = False
    if hasattr(model, 'frame_idxs'):
        try:
            model.frame_idxs = set()
        except Exception:
            pass

    batch = make_dummy_batch(bs, c, H, W, t, device)
    outputs = {}
    _ = model.forward(batch, outputs, is_train=False, timestamp=t)

    return

if __name__ == '__main__':

    # model
    from pytorch_lightning import seed_everything

    seed_everything(43, workers=True)

    import torch
    torch.autograd.set_detect_anomaly(True)
    import sys
    import os.path as osp
    sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
    from config import get_cfg, get_parser
    from TemporalStereo import TemporalStereo

    args = get_parser().parse_args()
    args.device = 'cpu'

    cfg = get_cfg(args)
    Net = TemporalStereo(cfg.convert_to_dict())
    
    # benchmark_once(Net, H=512, W=960, bs=1, c=3, t=0, warmup=10, iters=50, amp=False)
    run_once(Net, H=512, W=960, bs=1, c=3, t=0, warmup=10, iters=50, amp=False)

    # total_flops,total_params = flops(Net,args.device)

    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")