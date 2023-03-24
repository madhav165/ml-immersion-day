import argparse
import numpy as np
import datetime
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn, loss, Trainer
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data import DataLoader, ArrayDataset
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet.gluon.contrib import estimator


def _get_data_loaders(batch_size, num_workers=8):
    """
    Method returns CIFAR10 data loaders.
    if data doesn't exist on local disk, it will be downloaded.
    """
    
    # Define data transformations
    transform_train = transforms.Compose([
    # Randomly crop an area and resize it to be 32x32
    transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
    # Randomly flip the image horizontally
    transforms.RandomFlipLeftRight(),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                         [0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.4914, 0.4822, 0.4465], 
                                                              [0.2023, 0.1994, 0.2010])])

    # Set train=True for training data
    # Set shuffle=True to shuffle the training data
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True).
                                       transform_first(transform_train),batch_size=batch_size,
                                       shuffle=True, last_batch='discard', num_workers=num_workers)

    # Set train=False for validation data
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False).
                                      transform_first(transform_test),batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)
    
    return train_data, test_data


def _get_resnet_model(device, num_classes):
    """
    Define a new net, which will "stich" existing pretrained 
    ResNet base and untrained output Dense layer
    """
    
    # Get trained ResNet from model zoo
    resnet_base = vision.resnet18_v1(pretrained=True, ctx = device) 
    # Create Dense layer will be output classifier
    dense_layer = gluon.nn.Dense(num_classes)
    # Randomly initialize weights of classifier layer
    dense_layer.initialize(mx.init.Xavier(), ctx=device)
    

    new_net = gluon.nn.HybridSequential()
    
    with new_net.name_scope():
        new_net.add(resnet_base.features)
        new_net.add(dense_layer)    
    
    new_net.hybridize()
    return new_net


def train(args):
    
    # Identify device (GPU or CPU) where training computation will happen
    device = mx.context.gpu() if mx.context.num_gpus()>0 else mx.context.cpu()
    
    train_data, test_data = _get_data_loaders(args.batch_size)
    resnet_based_model = _get_resnet_model(device, args.num_classes)
    
    # Define context of training: loss, optimizer, and validation metrics
    softmax_cross_entropy = loss.SoftmaxCrossEntropyLoss(sparse_label=True) 
    optimizer = mx.optimizer.RMSProp(learning_rate=args.learning_rate, gamma1=1e-6)
    metrics = [mx.metric.Accuracy(), mx.metric.Loss()]
    

    # Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
    trainer = Trainer(resnet_based_model.collect_params(), optimizer)
    est = estimator.Estimator(net=resnet_based_model,
                              loss=softmax_cross_entropy,
                              val_metrics=metrics,
                              trainer=trainer,
                              context=device)

    # start training loop
    est.fit(train_data=train_data, val_data=test_data, 
            epochs=args.num_epochs)



if __name__ == "__main__":
    
    # Handling training hyper parameters 
    parser = argparse.ArgumentParser(description='MXNet MNIST Distributed Example')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs (default: 25)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate (default: 0.01)')
    parser.add_argument('--num-classes', type=int, default=10, help="size of output dense layer, should be equal to number of classes (default: 10)")
    args = parser.parse_args()
     
    train(args)