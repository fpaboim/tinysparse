import os
import time
import numpy as np
from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from extra.utils import get_parameters, fetch
from tqdm import trange
import tinygrad.optim as optim
import io
import tarfile
import pickle

class TinyLinear:
  def __init__(self, classes=10):
    inter_chan, out_chan = 8, 16   # for speed
    self.l0 = Linear(32*32*3, 64)
    self.l1 = Linear(64, 32)
    self.l2 = Linear(32, 32)
    self.lo = Linear(32, classes)

  def forward(self, x):
    x = self.l0(x).relu()
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    x = self.lo(x).logsoftmax()
    return x

class TinyConvNet:
  def __init__(self, classes=10):
    conv = 3
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.uniform(inter_chan,3,conv,conv)
    self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.uniform(out_chan*6*6, classes)

  def forward(self, x):
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).logsoftmax()

def load_cifar():
  filename = '../data/cifar-10-python.tar.gz'
  tt = tarfile.open(filename)
  db = pickle.load(tt.extractfile('cifar-10-batches-py/data_batch_1'), encoding="bytes")
  X = db[b'data'].reshape((-1, 3, 32, 32))
  Y = np.array(db[b'labels'])
  return X, Y

if __name__ == "__main__":
  X_train, Y_train = load_cifar()
  classes = 10
  split = 0.9
  lim = int(split*len(X_train))

  X_val = X_train[lim:]
  X_train = X_train[:lim]
  Y_val = Y_train[lim:]
  Y_train = Y_train[:lim]

  Tensor.default_gpu = True
  TINY = True
  TRANSFER = False
  if TINY:
    # model = TinyConvNet(classes)
    model = TinyLinear(classes)
  elif TRANSFER:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=True)
    model.load_weights_from_torch()
  else:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=False)

  parameters = get_parameters(model)
  print("parameters", len(parameters))
  optimizer = optim.Adam(parameters, lr=0.001)

  #BS, steps = 16, 32
  BS = 512 * 1
  epochs = 100

  losses = []
  val_losses = []
  accs = []
  val_accs = []

  if X_train.shape[0]%BS == 0:
    train_batches = int(X_train.shape[0]/BS)
  else:
    train_batches = int(X_train.shape[0]/BS) + 1

  if X_val.shape[0]%BS == 0:
    val_batches = int(X_val.shape[0]/BS)
  else:
    val_batches = int(X_val.shape[0]/BS) + 1


  for i in range(epochs):
    print("\nEPOCH:", i+1)
    rand_idx = np.random.permutation(X_train.shape[0])

    for j in trange(train_batches):
      train_slice = rand_idx[j*BS:(j+1)*BS]
      img = X_train[train_slice].astype(np.float32)
      img = img.reshape([img.shape[0], -1])

      st = time.time()
      out = model.forward(Tensor(img))
      fp_time = (time.time()-st)*1000.0

      Y = Y_train[train_slice]
      y = np.zeros((len(train_slice),classes), np.float32)
      y[range(y.shape[0]),Y] = -classes
      y = Tensor(y)
      loss = out.logsoftmax().mul(y).mean()

      optimizer.zero_grad()

      st = time.time()
      loss.backward()
      bp_time = (time.time()-st)*1000.0

      st = time.time()
      optimizer.step()
      opt_time = (time.time()-st)*1000.0

      #print(out.cpu().data)

      st = time.time()
      loss = loss.cpu().data
      cat = np.argmax(out.cpu().data, axis=1)
      accuracy = (cat == Y).mean()
      finish_time = (time.time()-st)*1000.0

      losses.append(loss)
      accs.append(accuracy)
      del out, y, loss

    rand_idx_val = np.random.permutation(X_val.shape[0])
    for j in trange(val_batches):
      val_slice = rand_idx_val[j*BS:(j+1)*BS]
      img = X_val[val_slice].astype(np.float32)
      img = img.reshape([img.shape[0], -1])

      st = time.time()
      out = model.forward(Tensor(img))
      fp_time = (time.time()-st)*1000.0

      Y = Y_val[val_slice]
      y = np.zeros((len(val_slice),classes), np.float32)
      y[range(y.shape[0]),Y] = -classes
      y = Tensor(y)
      val_loss = out.logsoftmax().mul(y).mean()

      val_loss = val_loss.cpu().data
      cat = np.argmax(out.cpu().data, axis=1)
      val_accuracy = (cat == Y).mean()
      finish_time = (time.time()-st)*1000.0

      val_losses.append(val_loss)
      val_accs.append(val_accuracy)
      del out, y, val_loss

    loss_mean = np.array(losses).mean()
    val_loss_mean = np.array(val_losses).mean()
    accuracy = np.array(accs).mean()
    val_loss_accuracy = np.array(val_accs).mean()

    # printing
    print("loss %.2f accuracy %.2f -- valloss %.2f + valacc %.2f" %
      (loss_mean, accuracy, val_loss_mean, val_loss_accuracy))

