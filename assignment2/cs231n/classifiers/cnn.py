from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_channel, self.im_height, self.im_width = input_dim
        self.num_classes = num_classes

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        # Initialize weights and biases for the three-layer convolutional
        self.params['W1'] = weight_scale * np.random.randn(num_filters, self.num_channel, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters, self.dtype)

        # Calculate input dim of hidden layer
        # Width and height should be kept same pre and post conv layer
        height_post_conv = self.im_height - 1 + filter_size % 2
        width_post_conv = self.im_width - 1 + filter_size % 2
        height_post_pool = height_post_conv // 2
        width_post_pool = width_post_conv // 2

        # Image should be flattened as input of hidden affine layer
        input_dim_hidden = num_filters * height_post_pool * width_post_pool

        self.params['W2'] = weight_scale * np.random.randn(input_dim_hidden, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim, self.dtype)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, self.num_classes)
        self.params['b3'] = np.zeros(num_classes, self.dtype)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        num_samples = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        v1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        v2, cache2 = affine_relu_forward(v1.reshape(num_samples, -1), W2, b2)
        v3, cache3 = affine_forward(v2, W3, b3)
        scores = v3

        if y is None:
            return scores

        ############################################################################
        # Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        grads = {}
        loss, dScore = softmax_loss(scores, y)
        # L2 regularization
        loss += .5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        # Compute gradients
        dv3, grads['W3'], grads['b3'] = affine_backward(dScore, cache3)
        dv2, grads['W2'], grads['b2'] = affine_relu_backward(dv3, cache2)
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dv2.reshape(v1.shape), cache1)

        # Gradients for L2 regularization
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads
