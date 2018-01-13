from builtins import range
from fast_layers import max_pool_forward_reshape
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    reshaped_x = x.reshape(x.shape[0], -1)
    out = reshaped_x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(*x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.copy(x)
    out[x<0] = 0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = np.copy(dout), cache
    dx[x<0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # Compute sample_mean and sample_var and use them to form unit Gaussion
        sample_mean = np.sum(x, axis=0) / N
        sample_var = np.sum(np.square(x - sample_mean), axis=0) / N
        zero_centered_x = x - sample_mean
        sample_corrected_std = np.sqrt(sample_var + eps)
        normalized_x = zero_centered_x / sample_corrected_std
        # Use gamma and beta to scale and shift
        out = normalized_x * gamma + beta
        cache = (x, zero_centered_x, normalized_x, gamma, beta, sample_mean, sample_corrected_std)
        # Keep average running mean and running var, use them in test time
        running_mean = (1 - momentum) * running_mean + momentum * sample_mean
        running_var = (1 - momentum) * running_var + momentum * sample_var

    elif mode == 'test':
        out = np.copy(x)
        out -= running_mean
        out /= (np.sqrt(running_var + eps))
        out = out * gamma + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, zero_centered_x, normalized_x, gamma, beta, sample_mean, sample_corrected_std = cache
    dx, dgamma, dbeta = np.zeros_like(x), np.zeros_like(gamma), np.zeros_like(beta)
    N, D = x.shape
    sample_corrected_var = sample_corrected_std**2
    for i in np.arange(N):
        for j in np.arange(D):
            mean_j = sample_mean[j]
            var_j = sample_corrected_var[j]
            dxij = np.zeros_like(dx[:, j])
            dxij -= np.sqrt(var_j)
            dxij -= (x[i, j] - mean_j) * (x[:, j] - mean_j) / np.sqrt(var_j)
            dxij[i] += N * np.sqrt(var_j)
            dxij *= dout[i, j] * gamma[j] / (N * var_j)
            dx[:, j] += dxij

    dgamma = np.sum(normalized_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    x, zero_centered_x, normalized_x, gamma, beta, mean, corrected_std = cache
    N, D = x.shape
    mul = gamma / (N * corrected_std**2)
    scale = dout * mul
    dx = corrected_std * (N * scale - np.sum(scale, axis=0)) - \
        zero_centered_x * np.sum(zero_centered_x * dout, axis=0) * mul / corrected_std
    dgamma = np.sum(normalized_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask, out = None, None

    if mode == 'train':
        mask = (np.random.uniform(size=x.shape) < p) / p
        out = x * mask

    elif mode == 'test':
        out = np.copy(x)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    pad, stride = conv_param['pad'], conv_param['stride']
    num_train, num_channel, im_height, im_width = x.shape
    num_filter, _, filter_height, filter_width = w.shape

    # Ensure the output height and width are available
    assert (im_height + 2 * pad - filter_height) % stride == 0, 'Invalid conv parameters'
    assert (im_width + 2 * pad - filter_width) % stride == 0, 'Invalid conv parameters'

    out_height = 1 + (im_height + 2 * pad - filter_height) // stride
    out_width = 1 + (im_width + 2 * pad - filter_width) // stride

    out = np.zeros((num_train, num_filter, out_height, out_width), dtype=x.dtype)

    # Pad zeros
    x_padded = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant',
                          constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

    for i in np.arange(num_train):
        for j in np.arange(num_filter):
            for m in np.arange(out_height):
                for n in np.arange(out_width):
                    out[i, j, m, n] = np.sum(x_padded[i, :, m*stride:m*stride+filter_height,
                                             n*stride:n*stride+filter_width] * w[j, :]) + b[j]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    num_train, num_channel, im_height, im_width = x.shape
    num_filter, _, filter_height, filter_width = w.shape
    _, _, out_height, out_width = dout.shape
    pad, stride = conv_param['pad'], conv_param['stride']
    dx, dw, db = np.zeros_like(x, dtype=x.dtype), np.zeros_like(w, dtype=w.dtype), np.zeros_like(b, dtype=b.dtype)
    db = np.sum(dout, axis=(0, 2, 3))

    for i in np.arange(num_train):
        for j in np.arange(num_filter):
            for m in np.arange(out_height):
                for n in np.arange(out_width):
                    x_padded_t, x_padded_l = m*stride-pad, n*stride-pad
                    x_bound_t = max(0, x_padded_t)
                    x_bound_b = min(im_height, x_padded_t+filter_height)
                    x_bound_l = max(0, x_padded_l)
                    x_bound_r = min(im_width, x_padded_l+filter_width)
                    w_bound_t, w_bound_l = max(0, -x_padded_t), max(0, -x_padded_l)
                    w_bound_b, w_bound_r = min(filter_height, im_height-x_padded_t), min(filter_width, im_width-x_padded_l)
                    dx[i, :, x_bound_t:x_bound_b, x_bound_l:x_bound_r] += dout[i, j, m, n] * w[j, :, w_bound_t:w_bound_b, w_bound_l:w_bound_r]
                    dw[j, :, w_bound_t:w_bound_b, w_bound_l:w_bound_r] += dout[i, j, m, n] * x[i, :, x_bound_t:x_bound_b, x_bound_l:x_bound_r]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    """
    num_train, num_channel, im_height, im_width = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']    
    assert (im_height - pool_height) % stride == 0, 'Invalid pooling parameters'
    assert (im_width - pool_width) % stride == 0, 'Invalid pooling parameters'    
    out_height = 1 + (im_height - pool_height) // stride
    out_width = 1 + (im_width - pool_width) // stride
    out = np.zeros((num_train, num_channel, out_height, out_width), dtype=x.dtype)    
    for m in np.arange(out_height):
        for n in np.arange(out_width):
            out[:, :, m, n] = x[:, :, m * stride:m * stride + pool_height, n * stride:n * stride + pool_width].max((2, 3))    
    cache = (x, pool_param)
    return out, cache
    """
    return max_pool_forward_fast_myself(x, pool_param)


def max_pool_forward_fast_myself(x, pool_param):
    num_train, num_channel, im_height, im_width = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    assert (im_height - pool_height) % stride == 0, 'Invalid pooling parameters'
    assert (im_width - pool_width) % stride == 0, 'Invalid pooling parameters'

    out_height = 1 + (im_height - pool_height) // stride
    out_width = 1 + (im_width - pool_width) // stride
    pools_shape = (num_train, num_channel, out_height, out_width, pool_height, pool_width)
    pools_strides = x.itemsize * np.array((num_channel * im_height * im_width, im_height * im_width, im_width * stride, stride, im_width, 1))
    pools = np.lib.stride_tricks.as_strided(x, shape=pools_shape, strides=pools_strides)
    out = pools.max((4, 5))

    cache = (x, pools, pool_param)
    return out, cache


def max_pool_forward_fast_myself_modified(x, pool_param):
    num_train, num_channel, im_height, im_width = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    assert (im_height - pool_height) % stride == 0, 'Invalid pooling parameters'
    assert (im_width - pool_width) % stride == 0, 'Invalid pooling parameters'

    out_height = 1 + (im_height - pool_height) // stride
    out_width = 1 + (im_width - pool_width) // stride
    pools_shape = (num_train, num_channel, out_height, out_width, pool_height, pool_width)
    pools_strides = x.itemsize * np.array((num_channel * im_height * im_width, im_height * im_width, im_width * stride, stride, im_width, 1))
    pools = np.lib.stride_tricks.as_strided(x, shape=pools_shape, strides=pools_strides)
    pools = pools.reshape(num_train*num_channel*out_height*out_width, -1)
    indices = pools.argmax(1)
    out = pools[np.arange(pools.shape[0]), indices].reshape(num_train, num_channel, out_height, out_width)

    cache = (x, pools, indices, pool_param)
    return out, cache


def max_pool_forward_fast_myself_special(x, pool_param):
    return max_pool_forward_reshape(x, pool_param)


def max_pool_backward_fast_myself_special(dout, cache):
    """
    Modified version of fast_layers, improved performance about 60%
    Only when max_pool_forward_fast_myself_special is called
    Only available and efficient when stride == pool_width == pool_height
    :param dout:
    :param cache:
    :return:
    - dx: Gradient with respect to x
    """
    x, x_reshaped, out, pool_param = cache
    out_reshaped = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = out_reshaped == x_reshaped
    dx = dout[:, :, :, np.newaxis, :, np.newaxis] * mask
    dx /= np.sum(mask, axis=(3,5), keepdims=True)
    dx = dx.reshape(x.shape)
    return dx


def max_pool_backward_fast_myself_nonspecial(dout, cache):
    """
    max_pool_forward_fast_myself_modified is called
    Only efficient when General case (with overlap, width != height etc.)

    :param dout:
    :param cache:
    :return:
    - dx: Gradient with respect to x
    """
    x, pools, indices, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    num_train, num_channel, out_height, out_width = dout.shape
    _, _, im_height, im_width = x.shape
    dx = np.zeros((num_train * num_channel * out_height * out_width, im_height * im_width), dtype=x.dtype)
    pos = np.arange(len(indices))
    out_pos = pos % (out_height * out_width)
    out_h, out_w = out_pos // out_width, out_pos % out_width
    pool_h, pool_w = indices // pool_width,  indices % pool_width
    x_h, x_w = out_h * stride + pool_h, out_w * stride + pool_w
    x_pos = x_h * im_width + x_w
    dx[np.arange(dx.shape[0]), x_pos] = dout.flatten()
    dx = dx.reshape(num_train, num_channel, out_height*out_width, -1).transpose(0, 1, 3, 2).sum(3).reshape(num_train, num_channel, im_height, im_width)
    return dx


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Can only be used if max_pool_forward_fast_myself_modified is called.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pools, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    num_train, num_channel, out_height, out_width = dout.shape

    dx = np.zeros_like(x, dtype=x.dtype)
    for i in np.arange(num_train):
        for j in np.arange(num_channel):
            for m in np.arange(out_height):
                for n in np.arange(out_width):
                    pool_max_idx = pools[i, j, m, n].argmax()
                    pool_h, pool_w = pool_max_idx // pool_width, pool_max_idx % pool_width
                    st_h, st_w = np.array([m, n]) * stride
                    dx[i, j, st_h + pool_h, st_w + pool_w] += dout[i, j, m, n]
    return dx


def max_pool_forward_general(x, pool_param):
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    if pool_height == pool_width == stride:
        return max_pool_forward_fast_myself_special(x, pool_param)
    else:
        return max_pool_forward_fast_myself(x, pool_param)


def max_pool_backward_general(dout, cache):
    pool_param = cache[-1]
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    if pool_height == pool_width == stride:
        return max_pool_backward_fast_myself_special(dout, cache)
    else:
        # Bad performance occurs here, we should avoid using such pool parameters.
        return max_pool_backward_naive(dout, cache)


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
