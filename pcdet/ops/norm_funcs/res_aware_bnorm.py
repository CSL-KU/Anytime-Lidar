import torch

class AbstractResAwareBatchNorm(torch.nn.Module):
    def __init__(self):
        super(AbstractResAwareBatchNorm, self).__init__()
        self.cur_res_idx = 0
        self.layers = torch.nn.ModuleList()

    def forward(self, x):
        return self.layers[self.cur_res_idx](x)

    def setResIndex(self, idx):
        self.cur_res_idx = idx

class ResAwareBatchNorm1d(AbstractResAwareBatchNorm):
    def __init__(self, num_channels, num_resolutions, eps, momentum):
        super(ResAwareBatchNorm1d, self).__init__()
        for i in range(num_resolutions):
            self.layers.append(torch.nn.BatchNorm1d(num_channels, \
                    eps=eps, momentum=momentum))

class ResAwareBatchNorm2d(AbstractResAwareBatchNorm):
    def __init__(self, num_channels, num_resolutions, eps, momentum):
        super(ResAwareBatchNorm2d, self).__init__()
        for i in range(num_resolutions):
            self.layers.append(torch.nn.BatchNorm2d(num_channels, \
                    eps=eps, momentum=momentum))
