import numpy as np
import torch
import unittest
from tinygrad.densetensor import DenseTensor, DEFAULT_DEVICE
from tinygrad.sparsetensor import SparseTensor
from extra.gradcheck import numerical_jacobian, jacobian, gradcheck

x_init = np.random.randn(3,3).astype(np.float32)
U_init = np.random.randn(3,3).astype(np.float32)
V_init = np.random.randn(3,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinygrad(unittest.TestCase):

  def test_backward_pass(self):
    def test_tinygrad():
      x = SparseTensor(x_init)
      W = DenseTensor(W_init)
      m = DenseTensor(m_init)
      out = x.dot(W).relu()
      out = out.logsoftmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.cpu().data, x.grad.cpu().data, W.grad.cpu().data

    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.detach().numpy(), x.grad, W.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_backward_pass_diamond_model(self):
    def test_tinygrad():
      u = DenseTensor(U_init)
      v = DenseTensor(V_init)
      w = DenseTensor(W_init)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = out.logsoftmax()
      out = out.sum()
      out.backward()
      return out.cpu().data, u.cpu().grad.data, v.cpu().grad.data, w.cpu().grad.data

    def test_pytorch():
      u = torch.tensor(U_init, requires_grad=True)
      v = torch.tensor(V_init, requires_grad=True)
      w = torch.tensor(W_init, requires_grad=True)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.sum()
      out.backward()
      return out.detach().numpy(), u.grad, v.grad, w.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_dropout(self):
    DenseTensor.training = True
    n, rate = 1_000_000, 0.1
    w = DenseTensor.ones(n).dropout(rate)
    non_zeros = np.count_nonzero(w.cpu().data)
    expected = n * (1 - rate)
    np.testing.assert_allclose(non_zeros, expected, rtol=1e-3)

  @unittest.skipUnless(not DEFAULT_DEVICE, "float64 not supported on GPU")
  def test_jacobian(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    torch_x = torch.tensor(x, requires_grad=True)
    torch_W = torch.tensor(W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

    tiny_x = DenseTensor(x)
    tiny_W = DenseTensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()
    J = jacobian(tiny_func, tiny_x)
    NJ = numerical_jacobian(tiny_func, tiny_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-5)

  @unittest.skipUnless(not DEFAULT_DEVICE, "float64 not supported on GPU")
  def test_gradcheck(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    tiny_x = DenseTensor(x)
    tiny_W = DenseTensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()

    self.assertTrue(gradcheck(tiny_func, tiny_x))

    # coarse approx. since a "big" eps and the non-linearities of the model
    self.assertFalse(gradcheck(tiny_func, tiny_x, eps = 0.1))

if __name__ == '__main__':
  unittest.main()
