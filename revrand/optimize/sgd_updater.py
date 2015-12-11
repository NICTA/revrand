import numpy as np

class SGDUpdater:
    def __init__(self):
        self.name = 'SGD'

    def __call__(self, x, grad):
        return x - grad

    def name(self):
        return self.name

class AdaDelta(SGDUpdater):

    def __init__(self, rho=0.95, epsilon=1e-6):

        if rho < 0 or rho > 1:
            raise ValueError("Decay rate 'rho' must be between 0 and 1!")

        if epsilon <= 0:
            raise ValueError("Constant 'epsilon' must be > 0!")

        self.name = 'ADADELTA'
        self.rho = rho
        self.epsilon = epsilon
        self.Eg2 = 0
        self.Edx2 = 0

    def __call__(self, x, grad):
        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad**2
        dx = - grad * np.sqrt(self.Edx2 + self.epsilon) / np.sqrt(self.Eg2 + self.epsilon)
        self.Edx2 = self.rho * self.Edx2 + (1 - self.rho) * dx**2
        return x + dx

class AdaGrad(SGDUpdater):

    def __init__(self, x_shape, eta=1, epsilon=1e-6):

        if eta <= 0:
            raise ValueError("Learning rate 'eta' must be > 0!")

        if epsilon <= 0:
            raise ValueError("Constant 'epsilon' must be > 0!")

        self.name = 'ADAGRAD'
        self.eta = eta
        self.epsilon = epsilon
        self.g2_hist = np.zeros(x_shape)

    def __call__(self, x, grad):
        self.g2_hist += np.power(grad, 2)
        return x - self.eta * grad / (self.epsilon + np.sqrt(self.g2_hist))

class Momentum(SGDUpdater):

    def __init__(self, x_shape, rho=0.5, eta=0.01):

        if eta <= 0:
            raise ValueError("Learning rate 'eta' must be > 0!")

        if rho < 0 or rho > 1:
            raise ValueError("Decay rate 'rho' must be between 0 and 1!")

        self.name = 'Momentum'
        self.eta = eta
        self.rho = rho
        self.dx = np.zeros(x_shape)

    def __call__(self, x, grad):
        self.dx = self.rho * self.dx - self.eta * grad
        return x + self.dx
