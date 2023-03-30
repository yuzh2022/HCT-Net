import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate

OPS = {
    'Dynamic_convOPS':lambda c, stride, affine, dp: Dynamic_convOPS(c, c, affine=affine, dropout_rate=dp),
    'none': lambda c, stride, affine, dp: ZeroOp(c, c, stride=stride),
    'identity': lambda c, stride, affine, dp: IdentityOp(c, c, affine=affine,dropout_rate=dp),
    'cweight': lambda c, stride, affine, dp: CWeightOp(c, c, affine=affine, dropout_rate=dp),
    'dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine, dilation=2, dropout_rate=dp),
    'dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine, use_depthwise=True, dropout_rate=dp),
    'shuffle_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine,has_shuffle=True),
    'conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine),

    'down_Dynamic_convOPS':lambda c, stride, affine, dp: Dynamic_convOPS(c, c, stride=2, affine=affine, dropout_rate=dp),
    'avg_pool': lambda c, stride, affine, dp: PoolingOp(c, c, affine=affine, pool_type='avg'),
    'max_pool': lambda c, stride, affine, dp: PoolingOp(c, c, affine=affine,pool_type='max'),
    'down_cweight': lambda c, stride, affine, dp: CWeightOp(c, c, stride=2, affine=affine, dropout_rate=dp),
    'down_dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dilation=2, dropout_rate=dp),
    'down_dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, use_depthwise=True, dropout_rate=dp),
    'down_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dropout_rate=dp),

    'up_Dynamic_convOPS':lambda c, stride, affine, dp: Dynamic_convOPS(c, c, stride=2, affine=affine,use_transpose=True, dropout_rate=dp),
    'up_cweight': lambda c, stride, affine, dp: CWeightOp(c, c, stride=2, affine=affine,use_transpose=True, dropout_rate=dp),
    'up_dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine,use_depthwise=True, use_transpose=True, dropout_rate=dp),
    'up_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, use_transpose=True, dropout_rate=dp),
    'up_dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dilation=2,use_transpose=True,  dropout_rate=dp),
}



def consistent_dim(states):  #维度一致
    # handle the un-consistent dimension
    # zbabby
    # concatenate all meta-node to output along channels dimension
    h_max, w_max = 0, 0
    for ss in states:
        if h_max < ss.size()[2]:
            h_max = ss.size()[2]
        if w_max < ss.size()[3]:
            w_max = ss.size()[3]
    #return [F.upsample(ss, (h_max, w_max),align_corners=False) for ss in states]
    return [interpolate(ss, (h_max, w_max)) for ss in states]

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def shuffle_layer(x, groups):
    print("#####shuffle_layer is useful!!!#########")
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class AbstractOp(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class BaseOp(AbstractOp): #weight, norm, dropout

    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=True, affine=True,
                 act_func='relu', dropout_rate=0, ops_order='act_weight_norm' ):
        super(BaseOp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_norm = use_norm
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.norm_type = norm_type
        #print("parameter:",self.in_channels,self.out_channels,self.use_norm,self.act_func,self.dropout_rate,self.ops_order,self.norm_type)
        # batch norm, group norm, instance norm, layer norm
        if self.use_norm:
            # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
            # 16 channels for one group is best
            if self.norm_before_weight:
                group = 1 if in_channels % 16 != 0 else in_channels // 16
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, in_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(in_channels, affine=affine)
            else:
                group = 1 if out_channels % 16 != 0 else out_channels // 16  #2
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, out_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(out_channels, affine=affine)
        else:
            self.norm = None

        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None

        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def norm_before_weight(self):
        for op in self.ops_list:
            if op == 'norm':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return{
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_norm': self.use_norm,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @staticmethod
    def is_zero_ops():
        return False

    def get_flops(self, x):
        raise NotImplementedError

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'norm':
                if self.norm is not None:
                    x = self.norm(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x


class ConvOps(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1,groups=1,
                 bias=False, has_shuffle=False, use_transpose=False, output_padding=0, use_depthwise=False,
                 norm_type='gn', use_norm=True, affine=True, act_func='relu', dropout_rate=0, ops_order='act_weight_norm'):
        super(ConvOps, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size #3
        self.stride = stride #1
        self.dilation = dilation #1
        self.groups = groups #1
        self.bias = bias #False
        self.has_shuffle = has_shuffle #False
        self.use_transpose = use_transpose #False
        self.use_depthwise = use_depthwise #False
        self.output_padding = output_padding #0

        padding = get_same_padding(self.kernel_size) #1
        if isinstance(padding, int):
            padding *= self.dilation #1
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # 'kernel_size', 'stride', 'padding', 'dilation' can either be 'int' or 'tuple' of int
        if use_transpose:
            if use_depthwise: # 1. transpose depth-wise conv
                self.depth_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=padding, output_padding=self.output_padding, groups=in_channels, bias=self.bias)
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            groups=self.groups, bias=False)
            else: # 2. transpose conv
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                            stride=self.stride, padding=padding,
                            output_padding=self.output_padding, dilation=self.dilation, bias=self.bias) #3 32 3 1 1 0 1 False
        else:
            if use_depthwise: # 3. depth-wise conv
                self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=padding,
                        dilation=self.dilation, groups=in_channels, bias=False)
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            groups=self.groups, bias=False)
            else: # 4. conv
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                            stride=self.stride, padding=padding,
                            dilation=self.dilation, bias=False) #3 32 3 1 1 0 1 False

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        basic_str = 'Conv'
        basic_str = 'Dilation' + basic_str if self.dilation > 1 else basic_str
        basic_str = 'Depth' + basic_str if self.use_depthwise else basic_str
        basic_str = 'Group' + basic_str if self.groups > 1 else basic_str
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        basic_str = '%dx%d_' % (kernel_size[0], kernel_size[1]) + basic_str
        return basic_str

    @property
    def config(self):
        config = {
            'name': ConvOps.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            'depth_wise': self.use_depthwise,
            'transpose': self.use_transpose,
        }
        config.update(super(ConvOps, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return ConvOps(**config)

    def weight_call(self, x):
        if self.use_depthwise:
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)
        #print("groups:",self.groups)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

class CWeightOp(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, groups=None,
                 bias=False, has_shuffle=False, use_transpose=False,output_padding=0, norm_type='gn',
                 use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight'):
        super(CWeightOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_transpose = use_transpose
        self.output_padding = output_padding

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # `kernel_size`, `stride`, `padding`, `dilation`
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, out_channels),
            nn.Sigmoid()
        )
        if stride >= 2:
            if use_transpose:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding, output_padding=self.output_padding,
                                                bias=False)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=False)
            group = 1 if out_channels % 16 != 0 else out_channels // 16
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        basic_str = 'ChannelWeight'
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        return basic_str

    @staticmethod
    def build_from_config(config):
        return CWeightOp(**config)

    def weight_call(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        rst = self.norm(self.conv(x*y)) if self.stride >= 2 else x*y
        #print("output", rst.shape)
        return rst






class PoolingOp(BaseOp):

    def __init__(self, in_channels, out_channels, pool_type, kernel_size=2, stride=2,
                 norm_type='gn', use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight'):
        super(PoolingOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == 1:
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
        elif self.pool_type == 'max':
            self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        config = {
            'name': PoolingOp.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride
        }
        config.update(super(PoolingOp, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return PoolingOp(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    def weight_call(self, x):
        return self.pool(x)

class IdentityOp(BaseOp):

    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=False, affine=True,
                 act_func=None, dropout_rate=0, ops_order='act_weight_norm'):
        super(IdentityOp, self).__init__(in_channels, out_channels, norm_type,use_norm, affine,
                                          act_func, dropout_rate, ops_order)

    @property
    def unit_str(self):
        return 'Identity'

    @property
    def config(self):
        config = {
            'name': IdentityOp.__name__,
        }
        config.update(super(IdentityOp, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return IdentityOp(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    def weight_call(self, x):
        return x


class ZeroOp(BaseOp):
    def __init__(self, in_channels, out_channels, stride):
        super(ZeroOp, self).__init__(in_channels, out_channels)
        self.stride = stride

    @property
    def unit_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroOp.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroOp(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
        else:
            padding = torch.zeros(n, c, h, w)
        padding = torch.autograd.Variable(padding, requires_grad=False)
        return padding



class attention2d(nn.Module):
    def __init__(self, in_channels,out_channels,stride, output_padding, ratios, K, temperature, init_weight=True, use_transpose=False,affine=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.stride=stride
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels!=3:
            hidden_channels = int(in_channels*ratios)+1
        else:
            hidden_channels = K

        '''if stride>=2 :
            if use_transpose:
                self.fc1=nn.ConvTranspose2d(in_channels, hidden_channels, 1,stride=stride,padding=padding,output_padding=output_padding, bias=False)
                self.fc2 = nn.ConvTranspose2d(hidden_channels, K, 1,stride=stride,padding=padding,output_padding=output_padding, bias=True)
            else:
                self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=stride, padding=padding, bias=False)
                self.fc2 = nn.Conv2d(hidden_channels, K, 1,stride=stride,padding=padding, bias=True)
        else:
            self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=stride, padding=padding, bias=False)
            self.fc2 = nn.Conv2d(hidden_channels, K, 1,stride=stride,padding=padding, bias=True)
            # self.bn = nn.BatchNorm2d(hidden_planes)
            #group = 1 if out_channels % 16 != 0 else out_channels // 16
            #self.norm = nn.GroupNorm(group, out_channels, affine=affine)'''

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=stride, bias=False)
        self.fc2 = nn.Conv2d(hidden_channels, K, 1, stride=stride, bias=True)
        #group = 1 if out_channels % 16 != 0 else out_channels // 16
        #self.norm = nn.GroupNorm(group, out_channels, affine=affine)

        '''self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_channels, K, 1, bias=True)'''


        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        y = self.avgpool(x)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y).view(y.size(0), -1)
        #rst = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
        return F.softmax(y/self.temperature, 1)


class Dynamic_convOPS(BaseOp):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False, use_transpose=False, norm_type='gn',
                 use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight',
                 ratio=0.25,  output_padding=1,   K=4,temperature=34, init_weight=True):
        super(Dynamic_convOPS, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)
        assert in_channels%groups==0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_padding = output_padding
        #print("output_padding", output_padding)
        self.has_shuffle = has_shuffle
        self.use_transpose = use_transpose
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.padding = get_same_padding(self.kernel_size)
        if isinstance(self.padding, int):
            self.padding *= self.dilation
        else:
            self.padding[0] *= self.dilation
            self.padding[1] *= self.dilation

        self.attention = attention2d(in_channels,out_channels, stride, output_padding, ratio, K, temperature, use_transpose,affine)

        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)




        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()


    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        basic_str = 'DynamicWeight'
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        return basic_str

    @staticmethod
    def build_from_config(config):
        return Dynamic_convOPS(**config)

    def weight_call(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        #print("softmax_attention",softmax_attention.shape)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        #print("aggregate_weight",aggregate_weight.shape)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            if self.stride>=2:
                if self.use_transpose:
                    output = F.conv_transpose2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              output_padding=self.output_padding, groups=self.groups*batch_size)
                else:
                    output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                                      padding=self.output_padding,
                                      dilation=self.dilation, groups=self.groups * batch_size)
            else:
                output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.output_padding,
                                dilation=self.dilation, groups=self.groups*batch_size)
        else:
            if self.stride>=2:
                if self.use_transpose:
                    output = F.conv_transpose2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              output_padding=self.output_padding, groups=self.groups*batch_size)
                else:
                    output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                                      padding=self.output_padding,
                                      dilation=self.dilation, groups=self.groups * batch_size)
            else:
                output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.output_padding,
                                dilation=self.dilation, groups=self.groups * batch_size)
        #print("output",output.shape)

        output = output.view(batch_size, self.out_channels,output.size(-2), output.size(-1))
        #print("output", output.shape)
        #rst = self.norm(self.conv(x*output)) if self.stride >= 2 else x*output

        return output









