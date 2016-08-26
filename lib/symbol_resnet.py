import mxnet as mx

def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0, prefix='', name='', suffix='', no_bias=True):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad, no_bias=no_bias, name='%s%s%s' % (prefix, name, suffix))
        bn = mx.symbol.BatchNorm(data=conv, name='bn%s%s' % (name, suffix), use_global_stats=True, fix_gamma=True)
        act = mx.symbol.Activation(data = bn, act_type=act_type, name='%s%s%s_relu' % (prefix, name, suffix))
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad, no_bias=no_bias, name='%s%s%s' % (prefix, name, suffix))
        bn = mx.symbol.BatchNorm(data=conv, name='bn%s%s' % (name, suffix), use_global_stats=True, fix_gamma=True)
        return bn

def residual_factory(data, num_filter, dim_match, l1, l2, l2_str_type):
    if l2_str_type == 0:
        l2_str = chr(ord('a')+l2)
    elif l2_str_type == 1:
        l2_str = 'a' if l2==0 else 'b%d' % l2
    else:
        raise ValueError, 'l2_str_type must be 0 or 1'

    if l2 > 0:
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, act_type='relu', conv_type=0, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2a')
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=True, act_type='relu', conv_type=0, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2b')

        conv3 = conv_factory(data=conv2, num_filter=num_filter*4, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, conv_type=1, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2c')
        new_data = identity_data + conv3
        act = mx.symbol.Activation(data=new_data, act_type='relu', name='res%d%s_relu' % (l1, l2_str))
        return act
    else:
        if dim_match: # if dimension match
            conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, act_type='relu', conv_type=0, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2a')
            # adopt project method in the paper when dimension increased
            project_data = conv_factory(data=data, num_filter=num_filter*4, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, conv_type=1, prefix='res', name='%da_' % l1, suffix='branch1')
        else:
            conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), no_bias=True, act_type='relu', conv_type=0, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2a')
            project_data = conv_factory(data=data, num_filter=num_filter*4, kernel=(1,1), stride=(2,2), pad=(0,0), no_bias=True, conv_type=1, prefix='res', name='%da_' % l1, suffix='branch1')
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=True, act_type='relu', conv_type=0, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2b')
        conv3 = conv_factory(data=conv2, num_filter=num_filter*4, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, conv_type=1, prefix='res', name='%d%s_' % (l1, l2_str), suffix='branch2c')

        new_data = project_data + conv3
        act = mx.symbol.Activation(data=new_data, act_type='relu', name='res%d%s_relu' % (l1, l2_str))
        return act

def residual_net(data, n_list, num_filter_list, l2_str_type=0):
    for i in range(len(n_list)):
        n = n_list[i]
        num_filter = num_filter_list[i]
        for j in range(n):
            data = residual_factory(data=data, num_filter=num_filter, dim_match=(i==0), l1=i+2, l2=j, l2_str_type=0 if (i==0 or i==len(n_list)-1) else l2_str_type)

    return data

def get_resnet50(num_classes):
    data=mx.symbol.Variable(name='data')
    conv = mx.symbol.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad =(3,3), no_bias=False, name='conv1')
    bn = mx.symbol.BatchNorm(data=conv, name='bn_conv1', use_global_stats=True, fix_gamma=True)
    act = mx.symbol.Activation(data = bn, act_type='relu', name='conv1_relu')
    pool1 = mx.symbol.Pooling(data=act, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1')
    resnet = residual_net(pool1, [3, 4, 6, 3], [64, 128, 256, 512], l2_str_type=0) #
    pool2 = mx.symbol.Pooling(data=resnet, kernel=(7,7), stride=(1, 1), pool_type='avg', name='pool5')
    flatten = mx.symbol.Flatten(data=pool2, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1000')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='prob')
    return softmax

def get_resnet101(num_classes):
    data=mx.symbol.Variable(name='data')
    conv = mx.symbol.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad =(3,3), no_bias=True, name='conv1')
    bn = mx.symbol.BatchNorm(data=conv, name='bn_conv1', use_global_stats=True, fix_gamma=True)
    act = mx.symbol.Activation(data = bn, act_type='relu', name='conv1_relu')
    pool1 = mx.symbol.Pooling(data=act, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1')
    resnet = residual_net(pool1, [3, 4, 23, 3], [64, 128, 256, 512], l2_str_type=1) #
    pool2 = mx.symbol.Pooling(data=resnet, kernel=(7,7), stride=(1, 1), pool_type='avg', name='pool5')
    flatten = mx.symbol.Flatten(data=pool2, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1000')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='prob')
    return softmax

def get_resnet152(num_classes):
    data=mx.symbol.Variable(name='data')
    conv = mx.symbol.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad =(3,3), no_bias=True, name='conv1')
    bn = mx.symbol.BatchNorm(data=conv, name='bn_conv1', use_global_stats=True, fix_gamma=True)
    act = mx.symbol.Activation(data = bn, act_type='relu', name='conv1_relu')
    pool1 = mx.symbol.Pooling(data=act, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1')
    resnet = residual_net(pool1, [3, 8, 36, 3], [64, 128, 256, 512], l2_str_type=1) #
    pool2 = mx.symbol.Pooling(data=resnet, kernel=(7,7), stride=(1, 1), pool_type='avg', name='pool5')
    flatten = mx.symbol.Flatten(data=pool2, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1000')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='prob')
    return softmax
