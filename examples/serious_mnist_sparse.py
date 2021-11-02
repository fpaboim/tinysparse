#!/usr/bin/env python
#inspired by https://github.com/Matuzas77/MNIST-0.17/blob/master/MNIST_final_solution.ipynb
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

import numpy as np
from tinygrad.densetensor import  GPU, DenseTensor
from tinygrad.sparsetensor import SparseTensor
from tinygrad.nn import BatchNorm2D
from extra.utils import get_parameters
from test.test_mnist import fetch_mnist
from extra.training import train, evaluate, sparse_categorical_crossentropy
import tinygrad.optim as optim
from extra.augment import augment_img
GPU = os.getenv("GPU", None) is not None
QUICK = os.getenv("QUICK", None) is not None
DEBUG = os.getenv("DEBUG", None) is not None

GPU = True
if GPU:
  print("** USING GPU **")

class MLP:
  def __init__(self):
    # self.weight1 = DenseTensor.uniform(784,32)
    self.weight1 = SparseTensor.uniform(784,784, randsparsity=0.8)
    self.weight11 = DenseTensor.uniform(784,32)
    self.weight2 = DenseTensor.uniform(32,10)

  def parameters(self):
    if DEBUG: #keeping this for a moment
      pars = [par for par in get_parameters(self) if par.requires_grad]
      no_pars = 0
      for par in pars:
        print(par.shape)
        no_pars += np.prod(par.shape)
      print('no of parameters', no_pars)
      return pars
    else:
      return get_parameters(self)

  def save(self, filename):
    with open(filename+'.npy', 'wb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        print("PAR:", par)
        if par.is_sparse():
          np.save(f, par.cpu())
        else:
          np.save(f, par.cpu().data)

  def load(self, filename):
    with open(filename+'.npy', 'rb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        try:
          par.cpu().data[:] = np.load(f)
          if GPU:
            par.gpu()
        except:
          print('Could not load parameter', par)

  def forward(self, x):
    x = self.weight1.dot(x).relu()
    x = x.dot(self.weight11).relu()
    x = x.dot(self.weight2)
    return x.logsoftmax()


if __name__ == "__main__":
  # lrs = [1e-4, 1e-5] if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
  # epochss = [2, 1] if QUICK else [13, 3, 3, 1]
  BS = 128

  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y) #+ lmbd*(model.weight1.abs() + model.weight2.abs()).sum().cpu().data
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  steps = len(X_train)//BS
  np.random.seed(1337)
  if QUICK:
    steps = 1
    X_test, Y_test = X_test[:BS], Y_test[:BS]

  model = MLP()
  model

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

  lr = 0.001
  epochs = 100
  optimizer = optim.SGD(model.parameters(), lr=lr)
  X_train = X_train / 255
  X_test = X_test / 255
  for epoch in range(1,epochs+1):
    #first epoch without augmentation
    # X_aug = X_train if epoch == 1 else augment_img(X_train)
    train(model, X_train, Y_train, optimizer, steps=steps, lossfn=lossfn, BS=BS)
    accuracy = evaluate(model, X_test, Y_test, BS=BS)
    # model.save(f'examples/checkpoint{accuracy * 1e6:.0f}')
