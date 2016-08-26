import os
import sys
from skimage import io, transform
import caffe
import mxnet as mx
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'lib'))

if len(sys.argv) < 2:
    print 'usage: %s image-path <test-layer>' % sys.argv[0]
    sys.exit(-1)

RESNET_LAYER_NUM = 101
TEST_LAYER = 'prob' if len(sys.argv)<3 else sys.argv[2]

MEAN_FILE = 'mxnetmodel/ResNet_mean.npy'

CAFFE_NET = 'caffemodel/ResNet-%d-deploy.prototxt' % RESNET_LAYER_NUM
CAFFE_MODEL = 'caffemodel/ResNet-%d-model.caffemodel' % RESNET_LAYER_NUM

MXNET_NET = 'mxnetmodel/ResNet-%d.json' % RESNET_LAYER_NUM
MXNET_MODEL = 'mxnetmodel/ResNet-%d.params' % RESNET_LAYER_NUM

synset = [l.strip() for l in open(os.path.join(ROOT_DIR, 'data', 'synset.txt')).readlines()]

#  caffe
caffe.set_mode_cpu()
caffe_net = caffe.Net(CAFFE_NET, caffe.TEST)
caffe_net.copy_from(CAFFE_MODEL)


#  mxnet
dev = mx.cpu()
save_dict = mx.nd.load(MXNET_MODEL)
arg_params = {}
aux_params = {}
for k, v in save_dict.items():
    tp, name = k.split(':', 1)
    if tp == 'arg':
        arg_params[name] = v
    elif tp == 'aux':
        aux_params[name] = v
    else:
        raise ValueError

sym = mx.sym.load(MXNET_NET)
internals = sym.get_internals()
sym = internals[TEST_LAYER+'_output']
mxnet_net = mx.model.FeedForward(sym,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 ctx=dev,
                                 allow_extra_params=True,
                                 numpy_batch_size=1)


#  mean image
mean_img = np.load(MEAN_FILE)

def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_img
    normed_img.resize(1, 3, 224, 224)
    return normed_img

batch = PreprocessImage(sys.argv[1])

#  prediction of caffe
caffe_prob = caffe_net.forward_all(**{'data': batch})['prob'][0]
caffe_pred = np.argsort(caffe_prob)[::-1]

caffe_out = caffe_net.blobs[TEST_LAYER].data
mxnet_out = mxnet_net.predict(batch)
print 'Difference of layer %s = %.4f' % (TEST_LAYER, np.linalg.norm(caffe_out-mxnet_out)/caffe_out.size)

if TEST_LAYER == 'prob':
    #  prediction of mxnet
    mxnet_prob = mxnet_net.predict(batch)[0]
    mxnet_pred = np.argsort(mxnet_prob)[::-1]

    K = 5

    print '-------------------------------------------------------------'
    print 'Caffe\'s output of the top %d targets: ' % K
    for i in xrange(K):
        print 'rank %d(prob=%.4f): %s' % (i+1, caffe_prob[caffe_pred[i]], synset[caffe_pred[i]])

    print '-------------------------------------------------------------'
    print 'Mxnet\'s output of the top %d targets: ' % K
    for i in xrange(K):
        print 'rank %d(prob=%.4f): %s' % (i+1, mxnet_prob[mxnet_pred[i]], synset[mxnet_pred[i]])
    '''
    '''
