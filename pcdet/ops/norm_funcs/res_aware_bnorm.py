import torch
import numpy as np

class AbstractResAwareBatchNorm(torch.nn.Module):
    def __init__(self, res_divs = [1.0], resdiv_mask = [True]):
        super(AbstractResAwareBatchNorm, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.cur_layer = None
        self.res_divs = np.array(res_divs)
        self.resdiv_mask = np.array(resdiv_mask)

    def forward(self, x):
        return self.cur_layer(x)

    def setResIndex(self, idx):
        self.cur_layer = self.layers[idx]

    def interpolate(self, max_grid_l):
        num_bn_params = 4
        num_channels = len(self.layers[0].weight.data)
        X = np.power(max_grid_l / self.res_divs[self.resdiv_mask], 2)
        Y = np.empty((num_channels, num_bn_params, len(X)))
        fitted_curves = [list() for ch in range(num_channels)]
        for ch in range(num_channels):
            for li in range(len(self.layers)):
                Y[ch, 0, li] = self.layers[li].weight.data[ch]
                Y[ch, 1, li] = self.layers[li].bias.data[ch]
                Y[ch, 2, li] = self.layers[li].running_mean[ch]
                Y[ch, 3, li] = self.layers[li].running_var[ch]

            for param_idx in range(num_bn_params):
                y = Y[ch, param_idx, :]
                coeffs = np.polyfit(X, y, 2)
                fitted_curve = np.poly1d(coeffs)
                fitted_curves[ch].append(fitted_curve)

        new_layers = torch.nn.ModuleList()
        ltype = type(self.layers[0])
        eps, momentum = self.layers[0].eps, self.layers[0].momentum
        layer_itr = 0
        for li, m in enumerate(self.resdiv_mask):
            if m:
                new_layers.append(self.layers[layer_itr])
                layer_itr += 1
            else:
                l_new = ltype(num_channels, eps=eps, momentum=momentum)
                x = (max_grid_l / self.res_divs[li]) ** 2
                for ch in range(num_channels):
                    l_new.weight.data[ch]  = fitted_curves[ch][0](x)
                    l_new.bias.data[ch]    = fitted_curves[ch][1](x)
                    l_new.running_mean[ch] = fitted_curves[ch][2](x)
                    l_new.running_var[ch]  = fitted_curves[ch][3](x)
                new_layers.append(l_new.cuda())

        self.layers = new_layers

class ResAwareBatchNorm1d(AbstractResAwareBatchNorm):
    def __init__(self, num_channels, res_divs, resdiv_mask, eps, momentum):
        super(ResAwareBatchNorm1d, self).__init__(res_divs, resdiv_mask)
        for i in range(sum(resdiv_mask)):
            self.layers.append(torch.nn.BatchNorm1d(num_channels, \
                    eps=eps, momentum=momentum))

class ResAwareBatchNorm2d(AbstractResAwareBatchNorm):
    def __init__(self, num_channels, res_divs, resdiv_mask, eps, momentum):
        super(ResAwareBatchNorm2d, self).__init__(res_divs, resdiv_mask)
        for i in range(sum(resdiv_mask)):
            self.layers.append(torch.nn.BatchNorm2d(num_channels, \
                    eps=eps, momentum=momentum))
