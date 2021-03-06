# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from roi_align_cuda import *

class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)
        roi_align_forward_cuda(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        roi_align_backward_cuda(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 ):
        super(RoIAlign, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)

