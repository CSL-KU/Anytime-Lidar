import torch

class FnInstanceNorm(torch.nn.Module):
    def __init__(self, num_channels, eps, momentum):
        super(FnInstanceNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = torch.nn.functional.instance_norm(x, 
                running_mean=None, running_var=None,
                weight=None, bias=None,
                use_input_stats=True,
                momentum=self.momentum, eps=self.eps)

        return x
