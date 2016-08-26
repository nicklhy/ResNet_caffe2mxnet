## Introduction
This is a tool to convert the [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) from caffe model to mxnet model. The weights are directly copied from caffe network blobs.

## Notice
* There is a symbol generation code(lib/symbol\_resnet.py) which provide the full implementation to build ResNet 50, 101, 152. Be careful with some small differences between them(i.e. first conv's no\_bias is set differently in ResNet-50 from the other two).
* In order to run the code, you have to download the pre-trained ResNet in [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) and save them to caffemodel. The directory should be like this:
```
    caffemodel
    ├── ResNet-101-deploy.prototxt
    ├── ResNet-101-model.caffemodel
    ├── ResNet-152-deploy.prototxt
    ├── ResNet-152-model.caffemodel
    ├── ResNet-50-deploy.prototxt
    ├── ResNet-50-model.caffemodel
    └── ResNet_mean.binaryproto
```
* The prediction of the converted mxnet model is somehow slightly different from the original caffe model. The exact reason seems to be the difference of BN layer(not sure yet). You can use src/test\_predict.py to get the difference of the outputs from caffe and mxnet models.
* Kaiming He use BN+Scale in the original caffe model to implement the BN in paper(with learned gamma/beta). I just copied the parameters of them to a single BN operator in mxnet.
* If you find the exact reason about the output difference, please send me an email(nicklhy at gmail dot com).
