import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = .0
    dW = np.zeros_like(W)
    num_train, dim = X.shape
    num_classes = W.shape[1]
    for i in xrange(num_train):
        score = np.exp(X[i].dot(W))
        sum_score = np.sum(score)
        prob_score = score / sum_score
        for j in xrange(num_classes):
            # Correct label, compute log loss and contribute to loss
            if j == y[i]:
                loss -= np.log(prob_score[j])
                dW[:, j] += (prob_score[j] - 1) * X[i]
            # Incorrect score, no compute to log loss, contribute to gradient
            else:
                dW[:, j] += prob_score[j] * X[i]

    loss = loss / num_train + .5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    return loss, dW


def softmax_loss_vectorized(W=None, X=None, y=None, reg=None, scores=None):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = .0
    dW, score = None, None

    if scores is None:
        dW = np.zeros_like(W)
        num_train, dim = X.shape
        num_classes = W.shape[1]
        score = X.dot(W)
    else:
        num_train, num_classes = scores.shape
        score = scores

    exp_score = np.exp(score)
    sum_score = np.sum(exp_score, axis=1)
    prob_score = (exp_score.T / sum_score).T
    correct_label_prob_score = prob_score[np.arange(num_train), y]
    loss -= np.sum(np.log(correct_label_prob_score))
    loss = loss / num_train

    # Modify prob_score, substract correct label score by 1
    if W is not None:
        loss += .5 * reg * np.sum(W * W)
        prob_score[np.arange(num_train), y] -= 1
        dW = X.T.dot(prob_score) / num_train + reg * W
    return loss, dW
