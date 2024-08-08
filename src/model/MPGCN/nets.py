import torch
import logging
from torch import nn
from .blocks import Spatial_Basic_Block, Temporal_MultiScale_Block, ST_Person_Attention

class Input_Branch(nn.Module):
    def __init__(self, num_channel, A, use_att, **kwargs):
        super(Input_Branch, self).__init__()

        module_list = [
            Basic_Block(num_channel, 64, A, use_att, **kwargs),
            Basic_Block(64, 64, A, use_att, **kwargs),
            Basic_Block(64, 32, A, use_att, **kwargs)
        ]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V))
        for layer in self.layers:
            x = layer(x)

        return x

class MPGCN(nn.Module):
    def __init__(self, data_shape, num_class, A, **kwargs):
        super(MPGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.ModuleList([
            Input_Branch(num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [
            Basic_Block(32*num_input, 128, A, stride=2, **kwargs),
            Basic_Block(128, 128, A, **kwargs),
            Basic_Block(128, 128, A, **kwargs),
            Basic_Block(128, 256, A, stride=2, **kwargs),
            Basic_Block(256, 256, A, **kwargs),
            Basic_Block(256, 256, A, **kwargs)
        ]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        N, I, C, T, V, M = x.size()

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.main_stream:
            x = layer(x)

        # extract feature
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # output
        x = self.global_pooling(x)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fcn(x)

        return x, feature

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, use_att=True, stride=1, kernel_size=[9,2], **kwargs):
        super(Basic_Block, self).__init__()
        
        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        self.scn = Spatial_Basic_Block(in_channels, out_channels, max_graph_distance, A, **kwargs)
        self.tcn = Temporal_MultiScale_Block(out_channels, temporal_window_size, stride, **kwargs)
        if use_att:
            self.att = ST_Person_Attention(out_channels, **kwargs)
        else:
            self.att = lambda x: x
            

    def forward(self, x):
        return self.att(self.tcn(self.scn(x)))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, Basic_Block):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)
