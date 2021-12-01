# sorted in order of increasing complexity
from tinygrad.densetensor import DenseTensor
from tinygrad.sparsetensor import SparseTensor
import numpy as np

class Optimizer:
  def __init__(self, params):
    self.params = [x for x in params if x.requires_grad]

  def zero_grad(self):
    for param in self.params:
      param.grad = None

class SGDp(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr
    self.factor = 0.999
    self.decay = 1*self.factor
    self.iter = 0

  def step(self):
    self.iter += 1
    param = 0
    for t in self.params:
      param += 1
      # print('GRADt:', (t.grad.cpu().data*10).sum())
      if t.is_sparse():
        # print('GRAD:', t.grad.cpu().data)

        # self.lr = self.lr * self.factor
        t.updategrad(t.grad, -self.lr)
        t.scale(0.995)
        # if self.iter % 2 == 0:
        #   t.scale(0.998)
        if self.iter % 4 == 0:
          t.prune(0.0001)

        if self.iter % 16 == 0:
          np.random.seed(self.iter+param)
          t.reset()
        if t.should_accgrad:
          if not t.accgrad:
            t.accgrad = SparseTensor(np.zeros((t.shape[0], t.shape[1])), ellwidth=t.ellw, ellwidtht=t.ellwt)
          else:
            t.accgrad.updategradacc(t.grad, -self.lr)
            # t.accgrad.updategrad(t.grad, 1)

            # if self.iter % 2 == 0:
            #   t.accgrad.scale(0.99)
            # # if self.iter % 16 == 0:
            # t.accgrad.prune(0.002)

      else:
        # print("UPDATE GRAD", t.grad.cpu().data, self.lr)
        # self.decay = self.decay * self.factor
        grad_data = t.grad.cpu().data
        # print('grad:', grad_data[0][0], grad_data[0][1], grad_data[1][0], grad_data[-1][-1], grad_data.sum())
        # t -= t.grad * self.lr

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr
    self.factor = 0.999
    self.decay = 1*self.factor
    self.iter = 0

  def step(self):
    self.iter += 1
    for t in self.params:
      # print('GRADt:', (t.grad.cpu().data*10).sum())
      if t.is_sparse():
        # print('GRAD:', t.grad.cpu().data)

        # self.lr = self.lr * self.factor
        t.updategrad(t.grad, -self.lr)
        # if self.iter % 2 == 0:
        #   t.scale(0.998)
        # if self.iter % 4 == 0:
        #   t.prune(0.0001)
      else:
        # print("UPDATE GRAD", t.grad.cpu().data, self.lr)
        # self.decay = self.decay * self.factor
        grad_data = t.grad.cpu().data
        # print('grad:', grad_data[0][0], grad_data[0][1], grad_data[1][0], grad_data[-1][-1], grad_data.sum())
        t -= t.grad * self.lr

class RMSprop(Optimizer):
  def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
    super().__init__(params)
    self.lr, self.decay, self.eps = lr, decay, eps

    self.v = [DenseTensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1.0 - self.decay) * t.grad * t.grad
      t -= (t.grad * self.lr).div(self.v[i].sqrt() + self.eps)

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    self.m = [DenseTensor.zeros(*t.shape, device=params[0].device, requires_grad=False).gpu() for t in self.params]
    self.v = [DenseTensor.zeros(*t.shape, device=params[0].device, requires_grad=False).gpu() for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      # print("T:", t)
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * t.grad * t.grad
      t -= a * self.m[i].div(self.v[i].sqrt() + self.eps)
