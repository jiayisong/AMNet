import torch
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn

from mmdet.models.builder import BACKBONES


def dla_build_norm_layer(cfg, num_features):
    """Build normalization layer specially designed for DLANet.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.


    Returns:
        Function: Build normalization layer in mmcv.
    """
    cfg_ = cfg.copy()
    if cfg_['type'] == 'GN':
        if num_features % 32 == 0:
            return build_norm_layer(cfg_, num_features)
        else:
            assert 'num_groups' in cfg_
            cfg_['num_groups'] = cfg_['num_groups'] // 2
            return build_norm_layer(cfg_, num_features)
    else:
        return build_norm_layer(cfg_, num_features)


class BasicBlock(BaseModule):
    """BasicBlock in DLANet.

    Args:
        in_channels (int): Input feature channel.
        out_channels (int): Output feature channel.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        conv_cfg (dict): Dictionary to construct and config
            conv layer.
        stride (int, optional): Conv stride. Default: 1.
        dilation (int, optional): Conv dilation. Default: 1.
        init_cfg (dict, optional): Initialization config.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 stride=1,
                 dilation=1,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        self.conv1 = build_conv_layer(
            #dict(type='DCNv2Deterministic') if stride == 2 and out_channels >= 256 else None,
            conv_cfg,
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.bn1 = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.bn2 = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.stride = stride

    def forward(self, x, identity=None):
        """Forward function."""

        if identity is None:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class Root(BaseModule):
    """Root in DLANet.

    Args:
        in_channels (int): Input feature channel.
        out_channels (int): Output feature channel.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        conv_cfg (dict): Dictionary to construct and config
            conv layer.
        kernel_size (int): Size of convolution kernel.
        add_identity (bool): Whether to add identity in root.
        init_cfg (dict, optional): Initialization config.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 kernel_size,
                 add_identity,
                 init_cfg=None):
        super(Root, self).__init__(init_cfg)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.bn = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.add_identity = add_identity

    def forward(self, feat_list):
        """Forward function.

        Args:
            feat_list (list[torch.Tensor]): Output features from
                multiple layers.
        """
        children = feat_list
        y1 = torch.cat(feat_list, 1)
        y2 = self.conv(y1)
        x = self.bn(y2)
        if self.add_identity:
            x += children[0]
        x = self.relu(x)
        return x, y1


class Tree(BaseModule):
    """Tree in DLANet.

    Args:
        levels (int): The level of the tree.
        block (nn.Module): The block module in tree.
        in_channels: Input feature channel.
        out_channels: Output feature channel.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        conv_cfg (dict): Dictionary to construct and config
            conv layer.
        stride (int, optional): Convolution stride.
            Default: 1.
        level_root (bool, optional): whether belongs to the
            root layer.
        root_dim (int, optional): Root input feature channel.
        root_kernel_size (int, optional): Size of root
            convolution kernel. Default: 1.
        dilation (int, optional): Conv dilation. Default: 1.
        add_identity (bool, optional): Whether to add
            identity in root. Default: False.
        init_cfg (dict, optional): Initialization config.
            Default: None.
    """

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 stride=1,
                 level_root=False,
                 root_dim=None,
                 root_kernel_size=1,
                 dilation=1,
                 is_tree2=True,
                 add_identity=False,
                 init_cfg=None):
        super(Tree, self).__init__(init_cfg)
        if root_dim is None:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.root = Root(root_dim, out_channels, norm_cfg, conv_cfg,
                             root_kernel_size, add_identity)
            self.tree1 = block(
                in_channels,
                out_channels,
                norm_cfg,
                #dict(type='DCNv2Deterministic') if stride == 2 else None,
                conv_cfg,
                stride,
                dilation=dilation)
            self.tree2 = block(
                out_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                1,
                dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                stride,
                root_dim=None,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                is_tree2=False,
                add_identity=add_identity)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                norm_cfg,
                conv_cfg,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                is_tree2=is_tree2,
                add_identity=add_identity)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        self.is_tree2 = is_tree2
        if stride > 1:
            self.downsample = nn.MaxPool2d(kernel_size=stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    1,
                    stride=1,
                    bias=False),
                dla_build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x, identity=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        identity = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, identity=identity)
        if self.levels == 1:
            x2 = self.tree2(x1)
            feat_list = [x2, x1] + children
            x, y = self.root(feat_list)
            if self.is_tree2:
                # return feat_list, x
                return x, y
            else:
                return x
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
            return x


@BACKBONES.register_module()
class DLANet(BaseModule):
    r"""`DLA backbone <https://arxiv.org/abs/1707.06484>`_.

    Args:
        depth (int): Depth of DLA. Default: 34.
        in_channels (int, optional): Number of input image channels.
            Default: 3.
        norm_cfg (dict, optional): Dictionary to construct and config
            norm layer. Default: None.
        conv_cfg (dict, optional): Dictionary to construct and config
            conv layer. Default: None.
        layer_with_level_root (list[bool], optional): Whether to apply
            level_root in each DLA layer, this is only used for
            tree levels. Default: (False, True, True, True).
        with_identity_root (bool, optional): Whether to add identity
            in root layer. Default: False.
        pretrained (str, optional): model pretrained path.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization
            config dict. Default: None
    """
    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512)),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4, 5),
                 frozen_stages=-1,
                 norm_cfg=None,
                 conv_cfg=None,
                 layer_with_level_root=(False, True, True, True),
                 with_identity_root=False,
                 pretrained=None,
                 init_cfg=None):
        super(DLANet, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalida depth {depth} for DLA')

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]

        block, levels, channels = self.arch_settings[depth]
        self.channels = channels
        self.num_levels = len(levels)
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        assert max(out_indices) < self.num_levels
        self.base_layer = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                channels[0],
                7,
                stride=1,
                padding=3,
                bias=False),
            dla_build_norm_layer(norm_cfg, channels[0])[1],
            nn.ReLU(inplace=True))

        # DLANet first uses two conv layers then uses several
        # Tree layers
        for i in range(2):
            level_layer = self._make_conv_level(
                channels[0],
                channels[i],
                levels[i],
                norm_cfg,
                conv_cfg,
                stride=i + 1)
            layer_name = f'level{i}'
            self.add_module(layer_name, level_layer)

        for i in range(2, self.num_levels):
            dla_layer = Tree(
                levels[i],
                block,
                channels[i - 1],
                channels[i],
                norm_cfg,
                # dict(type='DCNv2') if i > 4 else None,
                conv_cfg,
                2,
                level_root=layer_with_level_root[i - 2],
                add_identity=with_identity_root)
            layer_name = f'level{i}'
            self.add_module(layer_name, dla_layer)

        self._freeze_stages()

    def _make_conv_level(self,
                         in_channels,
                         out_channels,
                         num_convs,
                         norm_cfg,
                         conv_cfg,
                         stride=1,
                         dilation=1):
        """Conv modules.

        Args:
            in_channels (int): Input feature channel.
            out_channels (int): Output feature channel.
            num_convs (int): Number of Conv module.
            norm_cfg (dict): Dictionary to construct and config
                norm layer.
            conv_cfg (dict): Dictionary to construct and config
                conv layer.
            stride (int, optional): Conv stride. Default: 1.
            dilation (int, optional): Conv dilation. Default: 1.
        """
        modules = []
        for i in range(num_convs):
            modules.extend([
                build_conv_layer(
                    # dict(type='DCNv2') if stride == 2 else None,
                    conv_cfg,
                    in_channels,
                    out_channels,
                    3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                dla_build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.base_layer.eval()
            for param in self.base_layer.parameters():
                param.requires_grad = False

            for i in range(2):
                m = getattr(self, f'level{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'level{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    # @profile
    def forward(self, x):
        outs = []
        x = self.base_layer(x)
        # print(f'basebn {self.base_layer[1].weight[0].item():.6f}')
        for i in range(self.num_levels):
            x = getattr(self, 'level{}'.format(i))(x)
            if isinstance(x, tuple):
                y = x[0]
                # print(y.shape)
                x = x[0]
            else:
                y = x
            if i in self.out_indices:
                outs.append(y)
        return tuple(outs)


