import os
import sys
import caffe
import mxnet as mx
#  import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'lib'))

RESNET_LAYER_NUM = 152

CAFFE_NET = 'caffemodel/ResNet-%d-deploy.prototxt' % RESNET_LAYER_NUM
CAFFE_MODEL = 'caffemodel/ResNet-%d-model.caffemodel' % RESNET_LAYER_NUM

MXNET_NET = 'mxnetmodel/ResNet-%d.json' % RESNET_LAYER_NUM
MXNET_MODEL = 'mxnetmodel/ResNet-%d.params' % RESNET_LAYER_NUM

import symbol_resnet

exec 'sym = symbol_resnet.get_resnet%d(1000)' % RESNET_LAYER_NUM

caffe_net = caffe.Net(CAFFE_NET, caffe.TEST)
caffe_net.copy_from(CAFFE_MODEL)

def copy_params(cnet, mnet):
    data_shape = cnet.blobs['data'].data.shape
    arg_shapes, output_shapes, aux_shapes = mnet.infer_shape(data=data_shape)
    arg_shape_map = dict(zip(mnet.list_arguments(), arg_shapes))
    aux_shape_map = dict(zip(mnet.list_auxiliary_states(), aux_shapes))
    arg_params_map = dict()
    aux_params_map = dict()
    for k, v in cnet.params.iteritems():
        if k.startswith('res') or k.startswith('conv') or k.startswith('fc'):
            assert (k+'_weight') in arg_shape_map
            assert v[0].data.shape == arg_shape_map[k+'_weight']
            #  if k == 'conv1':
                #  v[0].data[:, [0, 2], :, :] = v[0].data[:, [2, 0], :, :]
            arg_params_map[k+'_weight'] = mx.nd.array(v[0].data)
            if len(v) == 2:
                assert (k+'_bias') in arg_shape_map
                assert v[1].data.shape == arg_shape_map[k+'_bias']
                arg_params_map[k+'_bias'] = mx.nd.array(v[1].data)
        elif k.startswith('bn'):
            assert k.replace('bn', 'scale') in cnet.params
            assert (k+'_gamma') in arg_shape_map
            assert (k+'_beta') in arg_shape_map
            assert (k+'_moving_var') in aux_shape_map
            assert (k+'_moving_mean') in aux_shape_map
            assert aux_shape_map[k+'_moving_mean'] == v[0].data.shape
            assert aux_shape_map[k+'_moving_var'] == v[1].data.shape
            assert arg_shape_map[k+'_gamma'] == cnet.params[k.replace('bn', 'scale')][0].data.shape
            assert arg_shape_map[k+'_beta'] == cnet.params[k.replace('bn', 'scale')][1].data.shape
            aux_params_map[k+'_moving_mean'] = mx.nd.array(v[0].data)
            aux_params_map[k+'_moving_var'] = mx.nd.array(v[1].data)
            arg_params_map[k+'_gamma'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][0].data)
            arg_params_map[k+'_beta'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][1].data)
            if len(v)==3:
                assert v[2].data.size == 1 and v[2].data[0] == 1
        elif k.startswith('scale'):
            assert k.replace('scale', 'bn') in cnet.params
        else:
            raise ValueError

    return arg_params_map, aux_params_map

arg_params_map, aux_params_map = copy_params(caffe_net, sym)
assert set(arg_params_map.keys()+['data', 'prob_label']) == set(sym.list_arguments())
assert set(aux_params_map.keys()) == set(sym.list_auxiliary_states())

save_dict = {('arg:%s' % k) : v for k, v in arg_params_map.items()}
save_dict.update({('aux:%s' % k) : v for k, v in aux_params_map.items()})
mx.nd.save(MXNET_MODEL, save_dict)
sym.save(MXNET_NET)
print 'Save network definition to %s, parameters to %s' % (MXNET_NET, MXNET_MODEL)

