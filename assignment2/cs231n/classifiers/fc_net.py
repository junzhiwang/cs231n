from builtins import range, object
from layer_utils import softmax_loss, affine_relu_forward, affine_relu_backward, affine_bn_relu_forward, affine_bn_relu_backward
from layers import affine_forward, affine_backward
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim, dtype=float)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes, dtype=float)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']
        hidden_out, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(hidden_out, W2, b2)

        if y is None:
            return scores

        grads = {}
        loss, dScore = softmax_loss(scores, y)
        loss += .5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        dX2, grads['W2'], grads['b2'] = affine_backward(dScore, cache2)
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dX2, cache1)
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in np.arange(len(hidden_dims)):
            self.params['W%d' % (i+1)] = weight_scale * np.random.randn(
                input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i])
            self.params['b%d' % (i+1)] = weight_scale * np.zeros(hidden_dims[i], dtype=np.float32)
        self.params['W%d' % self.num_layers] = weight_scale * np.random.randn(
            hidden_dims[-1] if len(hidden_dims) > 0 else input_dim, num_classes)
        self.params['b%d' % self.num_layers] = weight_scale * np.zeros(num_classes, dtype=np.float32)

        if use_batchnorm:
            for i in np.arange(len(hidden_dims)):
                self.params['gamma%d' % (i+1)] = np.ones(hidden_dims[i])
                self.params['beta%d' % (i+1)] = np.zeros(hidden_dims[i])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        cache = self.num_layers * [None]
        for i in np.arange(self.num_layers-1):
            if not self.use_batchnorm:
                scores, cache[i] = affine_relu_forward(
                    X if i == 0 else scores, self.params['W%d' % (i+1)], self.params['b%d' % (i+1)])
            else:
                scores, cache[i] = affine_bn_relu_forward(X if i == 0 else scores, self.params['W%d' % (i+1)],
                                                          self.params['b%d' % (i+1)], self.params['gamma%d' % (i+1)],
                                                          self.params['beta%d' % (i+1)], self.bn_params[i])

        scores, cache[self.num_layers-1] = affine_forward(
            scores, self.params['W%d' % self.num_layers], self.params['b%d' % self.num_layers])
        ############################################################################
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dscore = softmax_loss(scores, y)
        dx, grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = affine_backward(
            dscore, cache[self.num_layers-1])

        for i in reversed(np.arange(self.num_layers-1)):
            if not self.use_batchnorm:
                dx, grads['W%d' % (i+1)], grads['b%d' % (i+1)] = affine_relu_backward(dx, cache[i])
            else:
                dx, grads['W%d' % (i+1)], grads['b%d' % (i+1)], grads['gamma%d' % (i+1)], grads['beta%d' % (i+1)] \
                    = affine_bn_relu_backward(dx, cache[i])

        for i in np.arange(self.num_layers):
            loss += .5 * self.reg * np.sum(np.square(self.params['W%d' % (i+1)]))
            grads['W%d' % (i+1)] += self.reg * self.params['W%d' % (i+1)]
        ############################################################################
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        ############################################################################

        return loss, grads
