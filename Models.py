import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import ws_resnet


class AGN(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(AGN, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.select_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        self.register_buffer('running_var', torch.zeros(1, num_channels, 1))

        self.register_buffer('fixed', torch.LongTensor([0]))

        self.select_weight_ = torch.cuda.FloatTensor([1., 1.])

        self.register_buffer("temperature", torch.ones((1,)).fill_(0.001))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.fixed.data.fill_(0)
        self.bias.data.zero_()
        self.temperature.data.fill_(0.001)

    def forward(self, x):
        x_group = x.view(1, x.size(0) * self.num_groups, -1)

        N, C, H, W = x.size()
        G = self.num_groups
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        temp = var_in + mean_in ** 2

        mean_gn = x_group.mean(-1, keepdim=True)
        var_gn = x_group.var(-1, keepdim=True)

        mean_gn = mean_gn.view(x.size(0), G, -1)
        var_gn = var_gn.view(x.size(0), G, -1)
        mean_gn = torch.stack([mean_gn] * (C // G), dim=2)
        var_gn = torch.stack([var_gn] * (C // G), dim=2)
        mean_gn = mean_gn.view(x.size(0), C, -1)
        var_gn = var_gn.view(x.size(0), C, -1)

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        if not self.fixed:
            self.select_weight_ = F.gumbel_softmax(self.select_weight, self.temperature[0], hard=False)
            if self.training and (max(self.select_weight_) - min(self.select_weight_) >= 1):
                self.fixed.data.fill_(1)
                self.select_weight.data = self.select_weight_.data
                self.select_weight_ = self.select_weight.detach()
        else:
            self.select_weight_ = self.select_weight.detach()

        mean = self.select_weight_[0] * mean_gn + self.select_weight_[1] * mean_bn
        var = self.select_weight_[0] * var_gn + self.select_weight_[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)

        return x * self.weight + self.bias

    def get_select(self):
        return self.select_weight_


def convert_resnet18_bn_to_gn(resnet18):
    # Change BN to GN
    resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

    resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
    resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
    resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
    resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

    resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
    resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
    resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
    resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
    resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

    resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
    resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
    resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
    resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
    resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

    resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
    resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
    resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
    resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
    resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

    assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'

    return resnet18


class client_model(nn.Module):
    def __init__(self, name, num_clients=None, pretrained=False):
        super(client_model, self).__init__()
        self.name = name
        self.num_clients = num_clients

        if 'LeNet' in self.name:
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

            if self.name.endswith('_bn'):
                self.bn1 = nn.BatchNorm2d(64)
                self.bn2 = nn.BatchNorm2d(64)

            elif self.name.endswith('_gn'):
                self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
                self.bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            elif self.name.endswith('_fednn'):
                self.conv1 = ws_resnet.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
                self.conv2 = ws_resnet.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
                self.bn1 = AGN(num_groups=2, num_channels=64)
                self.bn2 = AGN(num_groups=2, num_channels=64)

        elif 'ResNet18' in self.name:
            if self.name.endswith('_bn'):
                self.model = models.resnet18()

            elif self.name.endswith('_fednn'):
                def _Norm(num_features):
                    return AGN(num_groups=2, num_channels=num_features)

                self.model = ws_resnet.l_resnet18(norm_layer=_Norm)

            else:
                # gn as default
                self.model = convert_resnet18_bn_to_gn(models.resnet18())

            if pretrained:
                state_dict = models.resnet.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
                model_state_dict = self.model.state_dict()
                for k in model_state_dict:
                    if k in model_state_dict and k in state_dict:
                        if model_state_dict[k].shape != state_dict[k].shape:
                            del state_dict[k]
                self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if 'LeNet' in self.name:
            if self.name == 'LeNet':
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
            else:
                # with norm
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))

            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif 'ResNet18' in self.name:
            x = self.model(x)

        return x
