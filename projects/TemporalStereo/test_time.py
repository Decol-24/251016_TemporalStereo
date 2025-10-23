import torch
import argparse
import argparse
import torch
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

def flops(Net,device):
    Net = Net.to(device)
    bs=1, 
    c=3, 
    H=256, 
    W=512, 
    t=0,

    batch = make_dummy_batch(bs, c, H, W, t, device)
    outputs = {}

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net, (batch, outputs, False, t))   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params

def make_dummy_batch(bs=1, c=3, H=256, W=512, t=0, device='cuda'):
    batch = {}
    batch[('color_aug', t, 'l')] = torch.randn(bs, c, H, W, device=device)
    batch[('color_aug', t, 'r')] = torch.randn(bs, c, H, W, device=device)
    return batch

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
    
    benchmark_once(Net, H=512, W=960, bs=1, c=3, t=0, warmup=10, iters=50, amp=False)

    # total_flops,total_params = flops(Net,args.device)

    # print(avg_run_time)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")