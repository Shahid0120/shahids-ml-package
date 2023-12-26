import numpy as np
import matplotlib.pyplot as plt

# LinearModel must alway implement fit and predict
class LinearModel(object):
  def __init__(self, step_size=0.2, max_iter=100, eps=1e-5, theta_0=None, verbose=True):
    self.theta_0 = theta_0
    self.step_size = step_size
    self.max_iter = max_iter
    self.eps = eps
    self.verbose = verbose

  def fit(self, x, y):
    """
    Args:
        x: Training example inputs. Shape (m, n).
        y: Training example labels. Shape (m,).
    """
  
    raise NotImplementedError('Subclass of LinearModel must implement fit method.')

  def predict(self, x):
    """Make a prediction given new inputs x.

    Args:
        x: Inputs of shape (m, n).

    Returns:
        Outputs of shape (m,).
    """
    raise NotImplementedError('Subclass of LinearModel must implement predict method.')

