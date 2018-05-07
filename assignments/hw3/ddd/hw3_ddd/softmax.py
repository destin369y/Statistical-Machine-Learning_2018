import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  K = theta.shape[1]
  pro = np.zeros((m, K))
  thetaX = np.zeros((m, K))
  thetaX_sub = np.zeros((m, K))
  for i in xrange(m):
    for j in xrange(K):
      thetaX[i, j] = np.dot(X[i, :], theta[:, j])
    thetaX_sub[i, :] = thetaX[i, :] - np.max(thetaX[i, :])
    pro[i, :] = np.exp(thetaX_sub[i, :]) / np.sum(np.exp(thetaX_sub[i, :]))

  for i in xrange(m):
    for k in xrange(K):
      J += (y[i] == k) * np.log(pro[i, k])
  J = - J / m

  regu = 0.0
  for j in xrange(1, dim):
    for k in xrange(K):
      regu += theta[j, k]**2
  regu = regu * reg / 2.0 / m

  J += regu

  for k in xrange(K): # IMPORTANT! axis=0
    grad[:, k] = - 1.0 / m * np.sum([X[i] * ((y[i] == k) * 1 - pro[i, k]) for i in xrange(m)], axis = 0) 
    grad[:, k] += reg / m * theta[:, k]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  K = theta.shape[1]
  thetaX = np.dot(X, theta)
  thetaX_sub = np.exp(thetaX - np.max(thetaX, axis = 1, keepdims = True))
  pro = thetaX_sub / np.sum(thetaX_sub, axis = 1, keepdims = True)
  y_label = (y.reshape(-1, 1) == np.arange(K)) * 1
  J = - 1.0 / m * np.sum(np.multiply(y_label, np.log(pro)))
  J += reg / 2.0 / m * np.sum(theta**2)

  grad = - 1.0 / m * np.dot(X.T, y_label - pro) + reg / m * theta
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
