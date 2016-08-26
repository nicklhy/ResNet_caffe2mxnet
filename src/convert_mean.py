import sys
import caffe
import numpy as np

if len(sys.argv) != 3:
    print 'usage: %s mean.binaryproto mean.npy' % sys.argv[0]
    sys.exit(-1)

mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(sys.argv[1], 'rb').read())

mean_npy = caffe.io.blobproto_to_array(mean_blob).squeeze()
np.save(sys.argv[2], mean_npy)
