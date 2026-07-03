import torch
import torch.nn as nn
from torch.autograd.function import Function, InplaceFunction


# ==========================================================
# Binary Activation
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
        else:
            return (
                output.div(scale)
                .add_(1)
                .div_(2)
                .add_(torch.rand(output.size()).add(-0.5))
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
# Quantizer
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
                .add(torch.rand(output.size()).add(-0.5))
                .div(scale)
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


# ==========================================================
# Helper Functions
# ==========================================================

def binarized(input, quant_mode='det'):
    return Binarize.apply(input, quant_mode)


def quantize(input, quant_mode, numBits):
    return Quantize.apply(input, quant_mode, numBits)


# ==========================================================
# Loss Functions
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


class SqrtHingeLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):

        margin = 1.0

        output = margin - input.mul(target)
        output[output.le(0)] = 0

        ctx.save_for_backward(input, target)

        loss = output.mul(output).sum(0).sum(1).div(target.numel())

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        input, target = ctx.saved_tensors

        margin = 1.0

        output = margin - input.mul(target)
        output[output.le(0)] = 0

        grad_input = target.clone()
        grad_input.mul_(-2)
        grad_input.mul_(output)
        grad_input.mul_(output.ne(0).float())
        grad_input.div_(input.numel())

        return grad_input, grad_input


# ==========================================================
# Instrumented Binary Linear Layer
# ==========================================================

class BinarizeLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(BinarizeLinear, self).__init__(*args, **kwargs)

    def forward(self, input):

        # Save floating-point input
        self.input_float = input.detach().cpu().clone()

        # Binary activation
        if input.size(1) != 784:
            input_b = binarized(input)
        else:
            input_b = input

        self.input_binary = input_b.detach().cpu().clone()

        # Save floating-point weights
        self.weight_float = self.weight.detach().cpu().clone()

        # Binary weights
        weight_b = binarized(self.weight)

        self.weight_binary = weight_b.detach().cpu().clone()

        # Linear computation
        out = nn.functional.linear(input_b, weight_b)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        # Save output
        self.output_float = out.detach().cpu().clone()

        return out


# ==========================================================
# Instrumented Binary Convolution
# ==========================================================

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(BinarizeConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):

        # Save floating-point input
        self.input_float = input.detach().cpu().clone()

        # Binary activation
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input

        self.input_binary = input_b.detach().cpu().clone()

        # Save floating-point weights
        self.weight_float = self.weight.detach().cpu().clone()

        # Binary weights
        weight_b = binarized(self.weight)

        self.weight_binary = weight_b.detach().cpu().clone()

        # Convolution
        out = nn.functional.conv2d(
            input_b,
            weight_b,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        # Save output
        self.output_float = out.detach().cpu().clone()

        return out
