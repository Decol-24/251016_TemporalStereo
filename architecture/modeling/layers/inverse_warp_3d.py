import torch
import torch.nn.functional as F

def inverse_warp_3d(img, disp, padding_mode='zeros', disp_Y=None):
    """
    Args:
        img:                    (Tensor), the source image (where to sample pixels)
                                [B, C, H, W] or [B, C, D, H, W]
        disp:                   (Tensor), disparity map of the target image
                                [B, D, H, W]
        padding_mode:           (str), padding mode, default is zero padding
        disp_Y:                 (Tensor): disparity map of the target image along Y-axis, i.e., Height dimension
                                [B, D, H, W]

    Returns:
        projected_img:          (Tensor), source image warped to the target image
                                [B, C, D, H, W]
    """
    #根据视差扭曲图片，填充为0，额外考虑D维度
    #img通常是右图，disp通常为负值
    #img [1,128,5,64,120] 128是channel维度
    #disp [1,5,64,120]
    device = disp.device
    B, D, H, W = disp.shape

    if disp_Y is not None:
        assert disp.shape == disp_Y.shape, 'disparity map along x and y axis should have same shape!'
    if img.dim() == 4:
        _, C, iH, iW = img.shape
        img = img.unsqueeze(2).expand(B, C, D, iH, iW)
    elif img.dim() == 5:
        assert D == img.shape[2], 'The disparity number should be same between image and disparity map!'
    else:
        raise ValueError('image is only allowed with 4 or 5 dimensions, '
                         'but got {} dimensions!'.format(img.dim()))

    # get mesh grid for each dimension
    grid_d = torch.linspace(0, D - 1, D).view(1, D, 1, 1).expand(B, D, H, W).to(device)
    grid_h = torch.linspace(0, H - 1, H).view(1, 1, H, 1).expand(B, D, H, W).to(device)
    grid_w = torch.linspace(0, W - 1, W).view(1, 1, 1, W).expand(B, D, H, W).to(device)

    # shift the index of W dimension with disparity
    grid_w = grid_w + disp
    if disp_Y is not None:
        grid_h = grid_h + disp_Y

    # normalize the grid value into [-1, 1]; (0, D-1), (0, H-1), (0, W-1)
    grid_d = (grid_d / (D - 1) * 2) - 1
    grid_h = (grid_h / (H - 1) * 2) - 1
    grid_w = (grid_w / (W - 1) * 2) - 1

    # concatenate the grid_* to [B, D, H, W, 3]
    grid_d = grid_d.unsqueeze(4)
    grid_h = grid_h.unsqueeze(4)
    grid_w = grid_w.unsqueeze(4)
    grid = torch.cat((grid_w, grid_h, grid_d), 4)

    # [B, C, D, H, W]
    projected_img = F.grid_sample(img, grid, padding_mode=padding_mode, align_corners=True)

    return projected_img