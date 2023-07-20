import torch
from typing import *
from ..utils import Normalizer
import torch.nn.functional as F


class Resample2d(torch.nn.Module):
    """ Module to sample tensors using Spatial Transformer Networks. Caches multiple grids in GPU VRAM for speed.

    Examples
    -------
    model = MyOpticFlowNetwork()
    resampler = Resample2d(device='cuda:0')
    # flows map t0 -> t1
    flows = model(images)
    t0 = resampler(images, flows)

    model = MyStereoNetwork()
    resampler = Resample2d(device='cuda:0', horiz_only=True)
    disparity = model(left_images, right_images)
    """

    def __init__(self,
                 size: Union[tuple, list] = None,
                 device: Union[str, torch.device] = None,
                 horiz_only: bool = False,
                 num_grids: int = 5):
        """ Constructor for resampler.

        Parameters
        ----------
        size: tuple, list. shape: (2,)
            height and width of input tensors to sample
        fp16: bool
            if True, makes affine matrices in half precision
        device: str, torch.device
            device on which to store affine matrices and grids
        horiz_only: bool
            if True, only resample in the X dimension. Suitable for stereo matching problems
        num_grids: int
            Number of grids of different sizes to cache in GPU memory.

            A "Grid" is a tensor of shape H, W, 2, with (-1, -1) in the top-left to (1, 1) in the bottom-right. This
            is the location at which we will sample the input images. If we don't do anything to this grid, we will
            just return the original image. If we add our optic flow, we will sample the input image at at the locations
            specified by the optic flow.

            In many flow and stereo matching networks, images are warped at multiple resolutions, e.g.
            1/2, 1/4, 1/8, and 1/16 of the original resolution. To avoid making a new grid for sampling 4 times every
            time this is called, we will keep these 4 grids in memory. If there are more than `num_grids` resolutions,
            it will calculate and discard the top `num_grids` most frequently used grids.
        """
        super().__init__()
        if size is not None:
            assert (type(size) == tuple or type(size) == list)
        self.size = size

        # identity matrix
        self.base_mat = torch.Tensor([[1, 0, 0], [0, 1, 0]])

        self.device = device
        self.horiz_only = horiz_only
        self.num_grids = num_grids
        self.sizes = []
        self.grids = []
        self.uses = []

    def forward(self, images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """ resample `images` according to `flow`

        Parameters
        ----------
        images: torch.Tensor. shape (N, C, H, W)
            images
        flow: torch.Tensor. shape (N, 2, H, W) in the flow case or (N, 1, H, W) in the stereo matching case
            should be in ABSOLUTE PIXEL COORDINATES.
        Returns
        -------
        resampled: torch.Tensor (N, C, H, W)
            images sampled at their original locations PLUS the input flow
        """
        # for MPI-sintel, assumes t0 = Resample2d()(t1, flow)
        if self.size is not None:
            H, W = self.size
        else:
            H, W = flow.size(2), flow.size(3)
        # print(H,W)
        # images: NxCxHxW
        # flow: Bx2xHxW
        grid_size = [flow.size(0), 2, flow.size(2), flow.size(3)]
        if not hasattr(self, 'grids') or grid_size not in self.sizes:
            if len(self.sizes) >= self.num_grids:
                min_uses = min(self.uses)
                min_loc = self.uses.index(min_uses)
                del (self.uses[min_loc], self.grids[min_loc], self.sizes[min_loc])
            # make the affine mat match the batch size
            self.affine_mat = self.base_mat.repeat(images.size(0), 1, 1)

            # function outputs N,H,W,2. Permuted to N,2,H,W to match flow
            # 0-th channel is x sample locations, -1 in left column, 1 in right column
            # 1-th channel is y sample locations, -1 in first row, 1 in bottom row
            this_grid = F.affine_grid(self.affine_mat, images.shape, align_corners=False).permute(0, 3, 1, 2).to(
                self.device)
            this_size = [i for i in this_grid.size()]
            self.sizes.append(this_size)
            self.grids.append(this_grid)
            self.uses.append(0)
            # print(this_grid.shape)
        else:
            grid_loc = self.sizes.index(grid_size)
            this_grid = self.grids[grid_loc]
            self.uses[grid_loc] += 1

        # normalize flow
        # input should be in absolute pixel coordinates
        # this normalizes it so that a value of 2 would move a pixel all the way across the width or height
        # horiz_only: for stereo matching, Y values are always the same
        if self.horiz_only:
            # flow = flow[:, 0:1, :, :] / ((W - 1.0) / 2.0)
            flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              torch.zeros((flow.size(0), flow.size(1), H, W))], 1)
        else:
            # for optic flow matching: can be displaced in X or Y
            flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        # sample according to grid + flow
        return F.grid_sample(input=images, grid=(this_grid + flow).permute(0, 2, 3, 1),
                             mode='bilinear', padding_mode='border', align_corners=False)


class Reconstructor:
    def __init__(
            self,
            gpu_id: int,
            augs: Dict = None
            ):
        """
        Class for reconstructing images.

        Parameters
        ----------
        gpu_id: int
            GPU id being used for training.
        augs: Dict, default None
            Augmentations relevant to normalizing images in reconstructor.
        """
        device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        self.resampler = Resample2d(device=device)
        if 'normalization' in augs.keys():
            mean = list(augs["normalization"]["mean"])
            std = list(augs["normalization"]["std"])
        else:
            mean = None
            std = None

        self.normalizer = Normalizer(mean=mean, std=std)

    def reconstruct_images(self, image_batch: torch.Tensor, flows: Union[tuple, list]) -> Tuple[tuple, tuple, tuple]:
        # SSIM DOES NOT WORK WITH Z-SCORED IMAGES
        # requires images in the range [0,1]. So we have to denormalize for it to work!
        image_batch = self.normalizer.denormalize(image_batch)
        if image_batch.ndim == 4:
            N, C, H, W = image_batch.shape
            num_images = int(C / 3) - 1
            t0 = image_batch[:, :num_images * 3, ...].contiguous().view(N * num_images, 3, H, W)
            t1 = image_batch[:, 3:, ...].contiguous().view(N * num_images, 3, H, W)
        elif image_batch.ndim == 5:
            N, C, T, H, W = image_batch.shape
            num_images = T - 1
            t0 = image_batch[:, :, :num_images, ...]
            t0 = t0.transpose(1, 2).reshape(N * num_images, C, H, W)
            t1 = image_batch[:, :, 1:, ...]
            t1 = t1.transpose(1, 2).reshape(N * num_images, C, H, W)
        else:
            raise ValueError('unexpected batch shape: {}'.format(image_batch))

        reconstructed = []
        t1s = []
        t0s = []
        flows_reshaped = []
        for flow in flows:
            # upsampled_flow = F.interpolate(flow, (h,w), mode='bilinear', align_corners=False)
            if flow.ndim == 4:
                n, c, h, w = flow.size()
                flow = flow.view(N * num_images, 2, h, w)
            else:
                n, c, t, h, w = flow.shape
                flow = flow.transpose(1, 2).reshape(n * t, c, h, w)

            downsampled_t1 = F.interpolate(t1, (h, w), mode='bilinear', align_corners=False)
            downsampled_t0 = F.interpolate(t0, (h, w), mode='bilinear', align_corners=False)
            t0s.append(downsampled_t0)
            t1s.append(downsampled_t1)
            reconstructed.append(self.resampler(downsampled_t1, flow))
            del (downsampled_t1, downsampled_t0)
            flows_reshaped.append(flow)

        return tuple(t0s), tuple(reconstructed), tuple(flows_reshaped)

    def __call__(self, image_batch: torch.Tensor, flows: Union[tuple, list]) -> Tuple[tuple, tuple, tuple]:
        return self.reconstruct_images(image_batch, flows)
