from _linear_model import LinearModel
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(LinearModel):
  def fit(self, x, y, type_optimisation=None):
    """
    Args:
        x: Training example inputs. Shape (m, n).
        y: Training example labels. Shape (m,).
    """
    # check type of optimisation choosen
    if type_optimisation == "MBGD":
      # mini batch stochastic descent
      self.mini_batch_gradient_descent(x, y)
    elif type_optimisation == "Newton":
      # Newton Method
      self.newton_method(x, y)
    elif type_optimisation == "BGD":
      # batch gradient descent
       self.batch_gradient_descent(x, y)
    else:
      raise ValueError("Invalid type_optimization. Use 'MBGD', 'newton', or 'BG'")

    # using returned values calculate y^ vector between 0 and 1

  def batch_gradient_descent(self, x, y, max_iter, eps):
    # starts from random point
    m , n = x.shape

    # n x 1 matrix
    theta_0 = np.zeros(n)
    number_iter = 0

    while number_iter <= max_iter : # || loss function < epos
      # calculate new theta using hessian 
      pass 
  
    pass

  def newton_method(self, x, y):
    # starts from random x coordinate
    # calculate tangent line
    # calculates intercepts
    # loops until found the minimum point
    pass

  def mini_batch_gradient_descent(self, x, y):
    pass


  def predict(self, x):
    # Make a prediction given new inputs x.
    """
    Args:
        x: Inputs of shape (m, n).

    Returns:
        Outputs of shape (m,).
    """
    return 1 / (1 + np.exp(-x.dot(self.theta)))
