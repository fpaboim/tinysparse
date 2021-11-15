# sorted in order of increasing complexity
from tinygrad.densetensor import DenseTensor

class Optimizer:
  def __init__(self, params):
    self.params = [x for x in params if x.requires_grad]

  def zero_grad(self):
    for param in self.params:
      param.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr
    self.factor = 0.97
    self.decay = 1*self.factor

  def step(self):
    for t in self.params:
      # print('GRADt:', (t.grad.cpu().data*10).sum())
      if t.is_sparse():
        # print('GRAD:', t.grad.cpu().data)
        t.updategrad(t.grad, -self.lr)
        #t te_ctx = None
      else:
        # print("UPDATE GRAD", t.grad.cpu().data, self.lr)
        self.decay = self.decay * self.factor
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
