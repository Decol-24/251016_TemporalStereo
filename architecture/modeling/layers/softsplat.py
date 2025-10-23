#!/usr/bin/env python
import torch
import torch.nn as nn

def _bounds_mask(H, W, y, x):
    return (x >= 0) & (x < W) & (y >= 0) & (y < H)

def _flatten_idx(N, C, H, W, n, c, y, x):
    # all tensors of same shape; return flat indices into (N,C,H,W).view(-1)
    return (((n * C + c) * H) + y) * W + x

class _SoftsplatFunctionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, flow):
        """
        input: (N,C,H,W)
        flow:  (N,2,H,W), flow[:,0]=dx, flow[:,1]=dy
        returns output: (N,C,H,W)
        """
        assert input.dim() == 4 and flow.dim() == 4
        N, C, H, W = input.shape
        assert flow.shape == (N, 2, H, W)

        device = input.device
        dtype = input.dtype

        # base grid
        y_grid = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(N, 1, H, W)
        x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(N, 1, H, W)

        Xp = x_grid + flow[:, 0:1]  # (N,1,H,W)
        Yp = y_grid + flow[:, 1:1+1]  # (N,1,H,W)

        x0 = torch.floor(Xp)
        y0 = torch.floor(Yp)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        # weights (双线性分配到四邻)
        w_nw = (x1 - Xp) * (y1 - Yp)
        w_ne = (Xp - x0) * (y1 - Yp)
        w_sw = (x1 - Xp) * (Yp - y0)
        w_se = (Xp - x0) * (Yp - y0)

        # integer target coords
        x0i = x0.to(torch.int64); y0i = y0.to(torch.int64)
        x1i = x1.to(torch.int64); y1i = y1.to(torch.int64)

        # broadcast channel dim
        inp = input  # (N,C,H,W)
        # expand weights to (N,C,H,W)
        w_nw = w_nw.expand_as(inp); w_ne = w_ne.expand_as(inp)
        w_sw = w_sw.expand_as(inp); w_se = w_se.expand_as(inp)

        # prepare index tensors for each neighbor (N,C,H,W) in flat view
        n_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, C, H, W).to(torch.int64)
        c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(N, C, H, W).to(torch.int64)

        # masks
        m_nw = _bounds_mask(H, W, y0i, x0i)
        m_ne = _bounds_mask(H, W, y0i, x1i)
        m_sw = _bounds_mask(H, W, y1i, x0i)
        m_se = _bounds_mask(H, W, y1i, x1i)

        # flat indices
        idx_nw = _flatten_idx(N, C, H, W, n_idx, c_idx, y0i.clamp(0, H-1), x0i.clamp(0, W-1))
        idx_ne = _flatten_idx(N, C, H, W, n_idx, c_idx, y0i.clamp(0, H-1), x1i.clamp(0, W-1))
        idx_sw = _flatten_idx(N, C, H, W, n_idx, c_idx, y1i.clamp(0, H-1), x0i.clamp(0, W-1))
        idx_se = _flatten_idx(N, C, H, W, n_idx, c_idx, y1i.clamp(0, H-1), x1i.clamp(0, W-1))

        out = torch.zeros_like(inp).view(-1)

        def scatter_add(mask, idx, weight):
            if mask.any():
                v = (inp * weight)[mask]
                out.index_put_((idx[mask],), v, accumulate=True)

        scatter_add(m_nw, idx_nw.view(-1), w_nw.view(-1))
        scatter_add(m_ne, idx_ne.view(-1), w_ne.view(-1))
        scatter_add(m_sw, idx_sw.view(-1), w_sw.view(-1))
        scatter_add(m_se, idx_se.view(-1), w_se.view(-1))

        output = out.view(N, C, H, W)

        # save for backward
        ctx.save_for_backward(input, flow, Xp, Yp, x0i, y0i, x1i, y1i,
                              m_nw, m_ne, m_sw, m_se)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        """
        gradOutput: (N,C,H,W)
        returns (gradInput, gradFlow)
        """
        input, flow, Xp, Yp, x0i, y0i, x1i, y1i, m_nw, m_ne, m_sw, m_se = ctx.saved_tensors
        N, C, H, W = input.shape
        device = input.device
        dtype = input.dtype

        # weights again
        x0 = x0i.to(dtype); y0 = y0i.to(dtype)
        x1 = x1i.to(dtype); y1 = y1i.to(dtype)

        w_nw = (x1 - Xp) * (y1 - Yp)
        w_ne = (Xp - x0) * (y1 - Yp)
        w_sw = (x1 - Xp) * (Yp - y0)
        w_se = (Xp - x0) * (Yp - y0)

        # gradInput: 对应你的 kernel_Softsplat_updateGradInput
        def safe_gather(go, yy, xx, mask):
            # gather gradOutput at integer coords; out-of-bounds treated as 0
            yyc = yy.clamp(0, H-1); xxc = xx.clamp(0, W-1)
            gathered = go.gather(2, yyc.unsqueeze(1).expand(N, C, H, W)) \
                        .gather(3, xxc.unsqueeze(1).expand(N, C, H, W))
            return gathered * mask  # mask broadcast到(N,C,H,W)

        go = gradOutput
        gnw = safe_gather(go, y0i, x0i, m_nw)
        gne = safe_gather(go, y0i, x1i, m_ne)
        gsw = safe_gather(go, y1i, x0i, m_sw)
        gse = safe_gather(go, y1i, x1i, m_se)

        gradInput = gnw * w_nw + gne * w_ne + gsw * w_sw + gse * w_se

        # gradFlow: 对应你的 kernel_Softsplat_updateGradFlow
        # d w.r.t. dx (flow[:,0]) 和 dy (flow[:,1])
        # ∂nw/∂x = - (y1 - Yp), ∂ne/∂x = + (y1 - Yp), ∂sw/∂x = - (Yp - y0), ∂se/∂x = + (Yp - y0)
        # ∂nw/∂y = - (x1 - Xp), ∂ne/∂y = - (Xp - x0), ∂sw/∂y = + (x1 - Xp), ∂se/∂y = + (Xp - x0)
        dwnw_dx = -(y1 - Yp); dwne_dx = +(y1 - Yp); dwsw_dx = -(Yp - y0); dwse_dx = +(Yp - y0)
        dwnw_dy = -(x1 - Xp); dwne_dy = -(Xp - x0); dwsw_dy = +(x1 - Xp); dwse_dy = +(Xp - x0)

        # sum over channels: input * gradOutput_at_neighbor * dweight
        # 先把邻居的 gradOutput 取出来
        go_nw = gnw
        go_ne = gne
        go_sw = gsw
        go_se = gse

        inp = input  # (N,C,H,W)

        # 对 x 分量
        gx = (inp * (go_nw * dwnw_dx + go_ne * dwne_dx + go_sw * dwsw_dx + go_se * dwse_dx)).sum(dim=1, keepdim=False)
        # 对 y 分量
        gy = (inp * (go_nw * dwnw_dy + go_ne * dwne_dy + go_sw * dwsw_dy + go_se * dwse_dy)).sum(dim=1, keepdim=False)

        gradFlow = torch.stack([gx, gy], dim=1)  # (N,2,H,W)

        return gradInput, gradFlow


def softsplat_function_torch(input, flow):
    return _SoftsplatFunctionTorch.apply(input, flow)

# ---- Drop-in wrapper: keep original public API exactly the same ----
def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    """
    与原文件一致的接口：
    FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType) -> tenOutput
    tenInput: (N,C,H,W)
    tenFlow : (N,2,H,W)  dx, dy
    tenMetric: None 或 (N,1,H,W)
    strType: 'summation' | 'average' | 'linear' | 'softmax'
    """
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ['summation', 'average', 'linear', 'softmax']

    # 行为与原实现保持一致（先根据类型拼接，再softsplat，再按需归一化）
    if strType == 'average':
        tenInput = torch.cat(
            [tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])],
            dim=1
        )
    elif strType == 'linear':
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], dim=1)
    elif strType == 'softmax':
        e = tenMetric.exp()
        tenInput = torch.cat([tenInput * e, e], dim=1)

    # 与原文件一致：调用“内部实现类”的 apply（此处换成纯 PyTorch 的 Autograd 实现）
    tenOutput = _SoftsplatFunctionTorch.apply(tenInput.contiguous(), tenFlow.contiguous())

    if strType != 'summation':
        tenNormalize = tenOutput[:, -1:, :, :]
        tenOutput = tenOutput[:, :-1, :, :] / (tenNormalize + 1e-22)

    return tenOutput

# 为了最大限度兼容旧代码：导出同名类别名（如果你别处直接用了 _FunctionSoftsplat.apply 也不炸）
_FunctionSoftsplat = _SoftsplatFunctionTorch


# ===== 下面是你原来 Module 的无 cupy 版封装 =====
class ModuleSoftsplat(nn.Module):
    def __init__(self, strType: str):
        super().__init__()
        assert strType in ['summation', 'average', 'linear', 'softmax']
        self.strType = strType

    def forward(self, tenInput, tenFlow, tenMetric):
        assert (tenMetric is None) or (tenMetric.shape[1] == 1)
        strType = self.strType

        if strType == 'average':
            tenInput = torch.cat([tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])], 1)
        elif strType == 'linear':
            tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)
        elif strType == 'softmax':
            e = tenMetric.exp()
            tenInput = torch.cat([tenInput * e, e], 1)

        tenOutput = softsplat_function_torch(tenInput.contiguous(), tenFlow.contiguous())

        if strType != 'summation':
            tenNormalize = tenOutput[:, -1:, :, :]
            tenOutput = tenOutput[:, :-1, :, :] / (tenNormalize + 1e-22)

        return tenOutput