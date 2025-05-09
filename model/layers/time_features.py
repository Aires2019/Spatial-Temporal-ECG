import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Here the code places an extraction network framework for time features
'''

class MDSA(nn.Module):
    def __init__(self, channels, factor=8):
        super(MDSA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(factor, channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(channels, factor, 1, bias=False)

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, d = x.shape
        x = x.reshape(b, groups, -1, d)
        x = x.permute(0, 2, 1, 3)
        # flatten
        x = x.reshape(b, -1, d)

        return x

    def forward(self, x):
        b, c, d = x.size()
        group_x = x.reshape(b * self.groups, -1, d)  # b*g,c//g,d
        # 1D
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(group_x))))  # (b*g, c//g ,1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(group_x))))  # (b*g, c//g ,1)
        x1 = self.gn(group_x * (avg_out + max_out).sigmoid())  # b*g, c//g, d

        # 2D
        group_x_2d = group_x.unsqueeze(2)
        x2 = self.conv3x3(group_x_2d)  # b*g, c//g, h, w
        x22 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # b*g, 1, c//g
        weights = torch.matmul(x22, x1)
        out = (group_x * weights.sigmoid()).reshape(b, c, d)
        out = self.channel_shuffle(out, 2)
        return out


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class Res2Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=5, stride=1, downsample=None, groups=1, base_width=26,
                 dilation=1, scale=4, first_block=True, norm_layer=nn.BatchNorm1d,
                 atten=True):

        super(Res2Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups
        # print(width)

        self.atten = atten

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1
        if first_block:
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                                              padding=kernel_size // 2, groups=1, bias=False, dilation=1)
                                    for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])
        self.first_block = first_block
        self.scale = scale

        self.conv3 = conv1x1(width * scale, planes * self.expansion)

        self.relu = Mish()
        self.bn3 = norm_layer(planes * self.expansion)  # bn reverse

        # self.dropout = nn.Dropout(.1)

        if self.atten is True:
            self.attention = MDSA(planes * self.expansion)
        else:
            self.attention = None

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):

        out = self.conv1(x)

        out = self.relu(out)
        out = self.bn1(out)  # bn reverse
        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)
        # Initialize output as empty tensor for proper concatenation
        y = 0
        for idx, conv in enumerate(self.convs):
            # Add previous y-value
            if self.first_block:
                y = xs[idx]
            else:
                y += xs[idx]
            y = conv(y)
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y
        # Use last chunk as x1
        if self.scale > 1:
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)

        # out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.atten:
            out = self.attention(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class MTFE(nn.Module):

    def __init__(self, num_classes=5, hidden_dim=64, output_dim=128, input_channels=12, single_view=False):
        super(MTFE, self).__init__()

        self.single_view = single_view

        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=25, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = Mish()

        self.layer1 = Res2Block(inplanes=hidden_dim, planes=output_dim, kernel_size=15, stride=2, atten=True)

        self.layer2 = Res2Block(inplanes=output_dim, planes=output_dim, kernel_size=15, stride=2, atten=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if not self.single_view:
            self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.layer1(output)

        output = self.layer2(output)

        output = self.avgpool(output)

        output = output.view(output.size(0), -1)

        if not self.single_view:
            output = self.fc(output)

        return output
