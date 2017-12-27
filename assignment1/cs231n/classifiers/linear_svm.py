import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # https://zhuanlan.zhihu.com/p/21478575
                # Induction: Denote k as 0,1,...D-1
                # Denote loss[i, j] as the classify score on class_j at sample i
                # if margin <= 0, loss[i, j] is constant 0, derivative of loss at this pos with respect to any W is 0
                # Here we computed loss[i, j] = sum(X[i, k] * W[k, j]) - sum(X[i, k] * W[k, y[i]]) + 1
                # if loss[i, j] > 0 then the partial derivative of loss[i, j] with respect to W[k, j] is X[i, k]
                # else the partial derivative of loss[i, j] with respect to W[k, j] is 0
                # the partial derivative of loss[i, j] with respect to W[k,
                # y[i]] is -X[i, k]
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += .5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]
    score = np.dot(X, W)
    delta = 1
    correct_class_score = score[np.arange(num_train), y]
    margin = np.maximum(.0, score - correct_class_score[:, np.newaxis] + 1)
    margin[np.arange(num_train), y] = .0
    loss = np.sum(margin) / num_train + .5 * reg * np.sum(W * W)

    # https://zhuanlan.zhihu.com/p/21478575
    margin[margin > 0] = 1.0
    incorrect_cnt = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] = -incorrect_cnt
    dW = np.dot(X.transpose(), margin) / num_train + reg * W
    return loss, dW
