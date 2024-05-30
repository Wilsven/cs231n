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
        # Calculate scores for each sample
        scores = X[i].dot(W)  # (D,) @ (D, C) -> (C,)
        # For numerical stability
        scores -= scores.max()
        # Exponentiate
        scores_exp = np.exp(scores)
        # Normalize
        softmax = scores_exp / scores_exp.sum()
        # Add cross entropies as loss for each true label
        loss -= np.log(softmax[y[i]])

        ##############################################
        #                  GRADIENT                  #
        ##############################################
        for j in range(num_classes):
            if j == y[i]:
                continue
            # For each incorrect label
            dW[:, j] += softmax[j] * X[i]
        # For each true label
        dW[:, y[i]] -= (1 - softmax[y[i]]) * X[i]

    # Average over number of training examples
    loss /= num_train

    # Add regularization to the loss
    loss += reg * np.sum(W**2)

    dW /= num_train  # average over number of training examples
    dW += 2 * reg * W  # add partial derivative of regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)  # (D,) @ (D, C) -> (C,)

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
    # Calculate scores for all samples
    scores = X @ W  # (N, D) @ (D, C) -> (N, C)
    # For numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)  # (N, C)
    # Exponentiate
    scores_exp = np.exp(scores)  # (N, C)
    # Normalize
    softmax = scores_exp / scores_exp.sum(axis=1, keepdims=True)
    # For true labels
    true_softmax = softmax[range(num_train), y].reshape(-1, 1)  # (N, 1)
    # Sum cross entropies for true labels as loss
    loss -= np.log(true_softmax).sum()
    # Average over number of training examples
    loss /= num_train
    # Add regularization to the loss
    loss += reg * np.sum(W**2)

    ##############################################
    #                  GRADIENT                  #
    ##############################################
    softmax[range(num_train), y] -= 1
    # Calculates gradient for all incorrect and true labels
    dW = X.T @ softmax  # (D, N) @ (N, C) -> (D, C)
    # Average over number of training examples
    dW /= num_train
    # Add partial derivative of regularization term
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
