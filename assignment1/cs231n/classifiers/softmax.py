from builtins import range
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
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        ##############################################
        #                    LOSS                    #
        ##############################################
        # calculate scores for each sample -> shape: (num_classes, )
        scores = X[i].dot(W)
        # for numerical stability
        scores -= scores.max()
        # exponentiate
        scores_exp = np.exp(scores)
        # normalize
        softmax = scores_exp / scores_exp.sum()
        # add cross entropies as loss for each true label
        loss -= np.log(softmax[y[i]])

        ##############################################
        #                  GRADIENT                  #
        ##############################################
        for j in range(num_classes):
            if j == y[i]:
                continue
            # for each incorrect label
            dW[:, j] += softmax[j] * X[i]
        # for each true label
        dW[:, y[i]] -= (1 - softmax[y[i]]) * X[i]

    # average loss
    loss /= num_train
    # and regularize
    loss += reg * np.sum(W * W)

    # average gradient
    dW /= num_train
    # add regularization
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    ##############################################
    #                    LOSS                    #
    ##############################################
    # calculate scores for all sample -> shape: (mini_batch, num_classes)
    scores = X.dot(W)
    # for numerical stability
    scores -= scores.max()
    # exponentiate
    scores_exp = np.exp(scores)
    # normalize
    softmax = scores_exp / scores_exp.sum(axis=1, keepdims=True)
    # for true labels -> shape: (mini_batch, 1)
    true_softmax = softmax[range(num_train), y].reshape(-1, 1)
    # sum cross entropies for true labels as loss
    loss -= np.log(true_softmax).sum()
    # average loss
    loss /= num_train
    # and regularize
    loss += reg * np.sum(W * W)

    ##############################################
    #                  GRADIENT                  #
    ##############################################
    softmax[range(num_train), y] -= 1
    dW = X.T.dot(softmax)
    # average gradient
    dW /= num_train
    # add regularization
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
