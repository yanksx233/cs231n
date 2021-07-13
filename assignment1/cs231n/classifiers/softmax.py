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

    N, C = X.shape[0], W.shape[1]

    score = X @ W   # (N, C)
    for i in range(N):

        score[i] -= np.max(score[i]) # 数值稳定

        f_yi = score[i, y[i]]
        e_fi = np.exp(score[i])
        sum_e_fi = np.sum(e_fi)
        
        loss += -f_yi + np.log(sum_e_fi)

        dW[:, y[i]] += -X[i]
        
        for j in range(C):
            dW[:, j] += e_fi[j] / sum_e_fi * X[i]

    loss /= N
    dW /= N
    if reg > 0:
        loss += reg * np.sum(W*W)
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

    N, C = X.shape[0], W.shape[1]

    score = X @ W   # (N, C)
    score -= np.max(score, axis=1, keepdims=True)
    score = np.exp(score)
    score_sum = np.sum(score, axis=1)

    loss = -np.log(score[range(N), y] / score_sum)
    loss = np.sum(loss) / N

    temp = np.zeros(score.shape)
    temp[range(N), y] = -1
    temp += score / score_sum[:, np.newaxis]
    dW = X.T @ temp
    dW /= N

    if reg > 0:
        loss += reg * np.sum(W*W)
        dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
