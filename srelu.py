import torch.nn as nn
import torch
import torch.nn.functional as Function
from torch.autograd import Variable

# class SRelu(nn.Module):
#     def __init__(self, alpha=0.1):
#         super(SRelu, self).__init__()
#         self.alpha = alpha
#
#     def forward(self, input):
#         return torch.div(torch.sqrt(input * input + self.alpha) + input, 2.0)


# class my_srelu(Function):
#
#     @staticmethod
#     def forward(ctx, input):
#         output = torch.div(torch.sqrt(input * input + 0.1) + input, 2.0)
#         ctx.save_for_backward(input)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.save_tensors
#         grad_input = (torch.div(input, 2.0 * torch.sqrt(input * input + 0.1)) + 0.5)
#         return grad_output * grad_input


class SRelu(nn.Module):
    def __init__(self, alpha=0.1):
        super(SRelu, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        alpha = self.alpha
        output = torch.div(torch.add(torch.sqrt(torch.add(torch.pow(input, 2), alpha)), input), 2.0)
        return output


if __name__ == "__main__":
    input = Variable(torch.randn(20, 10)/100.0, requires_grad=True)
    srelu = SRelu()
    output = srelu(input).sum()
    output.backward()
    print("input: ", input)
    print(input.grad)
    # print("srelu: ", srelu(input))