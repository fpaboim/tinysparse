import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

import numpy as np
from tinygrad.densetensor import DenseTensor, GPU
from tinygrad.sparsetensor import SparseTensor
from tinygrad.nn import BatchNorm2D
from extra.utils import get_parameters
from test.test_mnist import fetch_mnist
from extra.training import pretrain, train, evaluate, sparse_categorical_crossentropy
import tinygrad.optim as optim
from extra.augment import augment_img
GPU = True
QUICK = False
DEBUG = False

np.random.seed(42)

# import networkx as nx
# G = nx.generators.random_graphs.fast_gnp_random_graph(784, .8, seed=3)
# m = nx.to_numpy_matrix(G)[:,:10] / 100
# print(m.shape)
dummy = DenseTensor.uniform(784,784).cpu().data

def layernorm(x, sz, eps=1e-5):
  in_shape = x.shape
  x = x.reshape(shape=(-1, sz))
  layer_mean = x.mean(axis=(1,))
  y = (x - layer_mean.reshape(shape=[-1, 1]))
  layer_var = (y*y).mean(axis=(1,))
  ret = y.div(layer_var.add(eps).reshape(shape=[-1, 1]).sqrt())
  return ret.reshape(shape=in_shape)

def cat(tensors, dim=1):
  num_dims = len(tensors[0].shape)
  # So you can set dim=-1 for last dim
  if dim < 0:
      dim = dim + num_dims
  last_dim = num_dims-1

  # If dim is not last, transpose tensor so that the concat dim is last
  if dim != last_dim:
      order = np.arange(len(tensors[0].shape))
      order[dim] = last_dim
      order[last_dim] = dim
      tensors = [t.transpose(order=order) for t in tensors]

  sizes = [t.shape[-1] for t in tensors]
  # Size of result along concat dim
  total_size = sum(sizes)

  iden = DenseTensor.eye(total_size)
  # Start result as all zeros
  concat = DenseTensor.zeros(*tensors[0].shape[:-1], total_size)
  start = 0
  for i,t in enumerate(tensors):
      # dot each tensor with a slice of the identity so it has the same size as concat and it's values
      # are in the right spots with zeros everywhere else
      concat = concat.add(t.dot(iden[start:start+sizes[i], :]))
      start += sizes[i]

  if dim != last_dim:
      # Undo transpose
      concat = concat.transpose(order=order)
  return concat

class MLP:
  def __init__(self):
    topk = 64
    self.layers = []
    self.nlayers = 8
    for i in range(self.nlayers):
      np.random.seed(i+42)
      weight = SparseTensor.uniform(784,784, randsparsity=0.7, topkx=topk, topky=topk, should_accgrad=True, ellwidth=topk, ellwidtht=topk)
      # weight = DenseTensor.uniform(784,784)
      self.layers.append(weight)

    self.weightf = DenseTensor.uniform(784,10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    x0 = x
    xi = x
    for i in range(self.nlayers):
      xt = x
      x = x.dot(self.layers[i]).relu()
      x = x + xt + x0
      x = layernorm(x, 784)

    x = x.dot(self.weightf)
    x = x.logsoftmax()
    return x

class MLP2:
  def __init__(self):
    topk = 64
    self.layers = []
    self.nlayers = 8
    for i in range(self.nlayers):
      np.random.seed(i+42)
      weight = SparseTensor.uniform(784,784, randsparsity=0.75, topkx=topk, topky=topk, should_accgrad=True, ellwidth=topk, ellwidtht=topk)
      # weight = DenseTensor.uniform(784,784)
      self.layers.append(weight)

    self.weightf  = DenseTensor.uniform(784,10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    xi = x
    xf = []
    for i in range(self.nlayers):
      x = x.dot(self.layers[i]).relu()
      xf.append(x)

    x = cat(xf)

    x = x.dot(self.weightf0)
    x = x.dot(self.weightf)
    x = x.logsoftmax()
    return x

if __name__ == "__main__":
  lrs = [1e-4] #if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
  epochs = 100
  BS = 4

  PRETRAIN       = False
  PRETRAIN_STEPS = 128
  PRETRAIN_BS    = 1

  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y)
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train/255 #if epoch == 1 else augment_img(X_train)
  X_test = X_test/255

  steps = len(X_train)//BS
  if QUICK:
    steps = 1
    X_test, Y_test = X_test[:BS], Y_test[:BS]

  model = MLP()

  if len(sys.argv) > 1:
    try:
      model.load(sys.argv[1])
      print('Loaded weights "'+sys.argv[1]+'", evaluating...')
      evaluate(model, X_test, Y_test, BS=BS)
    except:
      print('could not load weights "'+sys.argv[1]+'".')

  if GPU:
    params = get_parameters(model)
    [x.gpu_() for x in params]

  optimizer = optim.SGD(model.parameters(), lr=.001)
  optimizerp = optim.SGDp(model.parameters(), lr=.001)

  if PRETRAIN:
    pretrain(model, X_train, Y_train, optimizerp, steps=len(X_train)//PRETRAIN_BS, BS=PRETRAIN_BS, pretrain_steps=PRETRAIN_STEPS)

    for weight in model.layers:
      accgrad = weight.accgrad.to_numpy()
      # accgrad *= 1.2
      weight.reset(accgrad)
      # weight.reset()
      # weight.updategradacc(weight.accgrad, 0.9)

    # model.weight1.updategrad(model.weight1.accgrad, -1)
  for epoch in range(1,epochs+1):
    #first epoch without augmentation
    train(model, X_train, Y_train, optimizer, steps=steps,  BS=BS)
    accuracy = evaluate(model, X_test, Y_test, BS=BS)
    # model.save(f'examples/checkpoint{accuracy * 1e6:.0f}')
