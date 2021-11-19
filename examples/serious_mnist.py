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
from extra.training import train, evaluate, sparse_categorical_crossentropy
import tinygrad.optim as optim
from extra.augment import augment_img
GPU = True
QUICK = False
DEBUG = False

class MLP:
  def __init__(self):
    # w_init = np.random.randn(784,10).astype(np.float32) / 1000
    # w_init2 = np.random.randn(BS,10).astype(np.float32) / 1000
    # self.weight1 = DenseTensor.uniform(784,10)
    self.weight1 = SparseTensor.uniform(784,10,randsparsity=0.01)
    # self.weight1 = SparseTensor.uniform(784,128,randsparsity=0.5)
    # self.weight2 = DenseTensor.uniform(784,10)
    # self.weight2 = SparseTensor.uniform(128,16)
    # self.weight2 = SparseTensor(w_init2)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    x = x.dot(self.weight1)
    # x = x.dot(self.weight2)
    x = x.logsoftmax()
    return x

if __name__ == "__main__":
  lrs = [1e-4] #if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
  epochs = 100
  BS = 32

  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y)
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  steps = len(X_train)//BS
  np.random.seed(1337)
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

  optimizer = optim.SGD(model.parameters(), lr=.0001)
  for epoch in range(1,epochs+1):
    #first epoch without augmentation
    X_aug = X_train #if epoch == 1 else augment_img(X_train)
    train(model, X_aug, Y_train, optimizer, steps=steps,  BS=BS)
    accuracy = evaluate(model, X_test, Y_test, BS=BS)
    # model.save(f'examples/checkpoint{accuracy * 1e6:.0f}')
