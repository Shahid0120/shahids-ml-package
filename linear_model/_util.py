import numpy as np
import matplotlib.pyplot as plt

# load-data set, ensure's it is numbers, creates design matrix and y and plots
class Util(object):
    def __init__(self, x, y, csv_path, label_col, theta, correction):
        self.x = x
        self.y = y
        self.csv_path = csv_path
        self.label_col = label_col
        self.theta = theta
        self.correction = correction

    def load_dataset(self):
      def add_intercept_fn(x):
        # Implement add_intercept logic here
        pass

      # Validate label_col argument
      allowed_label_cols = ('y', 't')
      if self.label_col not in allowed_label_cols:
          raise ValueError(f'Invalid label_col: {self.label_col} (expected {allowed_label_cols})')

      # Load headers
      with open(self.csv_path, 'r') as csv_fh:
          headers = csv_fh.readline().strip().split(',')

      # Load features and labels
      x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
      l_cols = [i for i in range(len(headers)) if headers[i] == self.label_col]
      inputs = np.loadtxt(self.csv_path, delimiter=',', skiprows=1, usecols=x_cols)
      labels = np.loadtxt(self.csv_path, delimiter=',', skiprows=1, usecols=l_cols)

      if inputs.ndim == 1:
          inputs = np.expand_dims(inputs, -1)

      if add_intercept:
          inputs = add_intercept_fn(inputs)
          pass

      return inputs, labels

    def add_intercept(self):
      new_x = np.zeros((self.x.shape[0], self.x.shape[1] + 1), dtype=self.x.dtype)
      new_x[:, 0] = 1
      new_x[:, 1:] = self.x
      return new_x

    def plot(self, save_path):
      """Plot dataset and fitted logistic regression parameters."""
      # Plot dataset
      plt.figure()
      plt.plot(self.x[self.y == 1, -2], self.x[self.y == 1, -1], 'bx', linewidth=2)
      plt.plot(self.x[self.y == 0, -2], self.x[self.y == 0, -1], 'go', linewidth=2)

      # Plot decision boundary (found by solving for theta^T x = 0)
      margin1 = (max(self.x[:, -2]) - min(self.x[:, -2])) * 0.2
      margin2 = (max(self.x[:, -1]) - min(self.x[:, -1])) * 0.2
      x1 = np.arange(min(self.x[:, -2]) - margin1, max(self.x[:, -2]) + margin1, 0.01)
      x2 = -(self.theta[0] / self.theta[2] * self.correction + self.theta[1] / self.theta[2] * x1)
      plt.plot(x1, x2, c='red', linewidth=2)
      plt.xlim(self.x[:, -2].min() - margin1, self.x[:, -2].max() + margin1)
      plt.ylim(self.x[:, -1].min() - margin2, self.x[:, -1].max() + margin2)

      # Add labels and save to disk
      plt.xlabel('x1')
      plt.ylabel('x2')
      plt.savefig(save_path)


