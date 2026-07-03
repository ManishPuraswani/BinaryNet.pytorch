import torch
import torch.nn as nn
import numpy as np
from torch.autograd.function import Function, InplaceFunction


# ==========================================================
# Binary Quantization
# ==========================================================

class Binarize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):

        ctx.inplace = inplace

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)

        return (
            output.div(scale)
                  .add_(1)
                  .div_(2)
                  .add_(torch.rand(output.size(), device=output.device).add(-0.5))
                  .clamp_(0, 1)
                  .round()
                  .mul_(2)
                  .add_(-1)
                  .mul(scale)
        )

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None


# ==========================================================
# Quantization
# ==========================================================

class Quantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, quant_mode='det', numBits=4, inplace=False):

        ctx.inplace = inplace

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = (2 ** numBits - 1) / (output.max() - output.min())

        output = output.mul(scale).clamp(
            -2 ** (numBits - 1) + 1,
            2 ** (numBits - 1)
        )

        if quant_mode == 'det':
            output = output.round().div(scale)
        else:
            output = (
                output.round()
                      .add(torch.rand(output.size(), device=output.device).add(-0.5))
                      .div(scale)
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None


def binarized(input, quant_mode='det'):
    return Binarize.apply(input, quant_mode)


def quantize(input, quant_mode='det', numBits=4):
    return Quantize.apply(input, quant_mode, numBits)


# ==========================================================
# Hinge Loss
# ==========================================================

class HingeLoss(nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):

        output = self.margin - input.mul(target)
        output[output.le(0)] = 0

        return output.mean()

    def forward(self, input, target):

        return self.hinge_loss(input, target)


# ==========================================================
# Square Hinge Loss
# ==========================================================

class SqrtHingeLossFunction(Function):

    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input, target):

        output = self.margin - input.mul(target)
        output[output.le(0)] = 0

        self.save_for_backward(input, target)

        loss = output.mul(output).sum(0).sum(1).div(target.numel())

        return loss

    def backward(self, grad_output):

        input, target = self.saved_tensors

        output = self.margin - input.mul(target)
        output[output.le(0)] = 0

        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())

        return grad_output, grad_output


# ==========================================================
# IMC Instrumented Binary Linear Layer
# ==========================================================

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):

        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

        # ---------------------------------------------
        # IMC Recorder
        # ---------------------------------------------

        self.record_tensors = False

        self.input_float = None
        self.input_binary = None

        self.weight_float = None
        self.weight_binary = None

        self.output_float = None

    def forward(self, input):

        # ---------------------------------------------
        # Save floating-point input
        # ---------------------------------------------

        if self.record_tensors:
            self.input_float = input.detach().cpu().clone()

        # ---------------------------------------------
        # Binarize input
        # ---------------------------------------------

        if input.size(1) != 784:
            input_b = binarized(input)
        else:
            input_b = input

        if self.record_tensors:
            self.input_binary = input_b.detach().cpu().clone()

        # ---------------------------------------------
        # Save floating weights
        # ---------------------------------------------

        if self.record_tensors:
            self.weight_float = self.weight.detach().cpu().clone()

        # ---------------------------------------------
        # Binarize weights
        # ---------------------------------------------

        weight_b = binarized(self.weight)

        if self.record_tensors:
            self.weight_binary = weight_b.detach().cpu().clone()

        # ---------------------------------------------
        # Linear Operation
        # ---------------------------------------------

        out = nn.functional.linear(
            input_b,
            weight_b
        )

        if self.bias is not None:

            self.bias.org = self.bias.data.clone()

            out += self.bias.view(1, -1).expand_as(out)

        # ---------------------------------------------
        # Save Output
        # ---------------------------------------------

        if self.record_tensors:
            self.output_float = out.detach().cpu().clone()

        return out

# ==========================================================
# IMC Instrumented Binary Convolution Layer
# ==========================================================

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):

        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

        # ---------------------------------------------
        # IMC Recorder
        # ---------------------------------------------

        self.record_tensors = False

        self.input_float = None
        self.input_binary = None

        self.weight_float = None
        self.weight_binary = None

        self.output_float = None

    def forward(self, input):

        # ---------------------------------------------
        # Save floating-point input
        # ---------------------------------------------

        if self.record_tensors:
            self.input_float = input.detach().cpu().clone()

        # ---------------------------------------------
        # Binarize input
        # (First RGB layer remains floating-point)
        # ---------------------------------------------

        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input

        if self.record_tensors:
            self.input_binary = input_b.detach().cpu().clone()

        # ---------------------------------------------
        # Save floating-point weights
        # ---------------------------------------------

        if self.record_tensors:
            self.weight_float = self.weight.detach().cpu().clone()

        # ---------------------------------------------
        # Binarize weights
        # ---------------------------------------------

        weight_b = binarized(self.weight)

        if self.record_tensors:
            self.weight_binary = weight_b.detach().cpu().clone()

        # ---------------------------------------------
        # Convolution
        # ---------------------------------------------

        out = nn.functional.conv2d(
            input_b,
            weight_b,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        # ---------------------------------------------
        # Bias
        # ---------------------------------------------

        if self.bias is not None:

            self.bias.org = self.bias.data.clone()

            out += self.bias.view(
                1,
                -1,
                1,
                1
            ).expand_as(out)

        # ---------------------------------------------
        # Save output
        # ---------------------------------------------

        if self.record_tensors:
            self.output_float = out.detach().cpu().clone()

        return out
