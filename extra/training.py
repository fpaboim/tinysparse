import os
import numpy as np
from tqdm import trange
from extra.utils import get_parameters
from tinygrad.densetensor import DenseTensor, GPU, Device
from tinygrad.sparsetensor import SparseTensor

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten()
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = DenseTensor(y)
  return out.mul(y).mean()

def pretrain(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy,
        transform=lambda x: x, target_transform=lambda x: x, pretrain_steps=64):
  DenseTensor.training = False
  SparseTensor.training = True
  losses, accuracies = [], []

  randperm = np.random.permutation(range(X_train.shape[0]))
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    if i > pretrain_steps:
      break
    samp = randperm[i*BS:(i+1)*BS]
    x = DenseTensor(transform(X_train[samp]))
    y = target_transform(Y_train[samp])

    # network
    out = model.forward(x)
    loss = lossfn(out, y)

    optim.zero_grad()
    loss.backward()
    # print('MODEL:', model.parameters())
    optim.step()

    cat = np.argmax(out.cpu().data, axis=-1)
    accuracy = (cat == y).mean()

    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    try:
      nnzs = 0
      gradnnzs = 0
      for weight in model.layers:
        nnzs += weight.count_nnzs()
        gradnnzs += weight.accgrad.count_nnzs()
        t.set_description("loss:%.2f  accuracy:%.2f  nnz:%i accgrad_nnz:%i" % (np.array(losses)[-4:].mean(), np.array(accuracies)[-4:].mean(), nnzs, gradnnzs))
      else:
        t.set_description("loss:%.2f  accuracy:%.2f  nnz:%i" % (np.array(losses)[-4:].mean(), np.array(accuracies)[-4:].mean(), nnzs))
    except Exception as e:
      t.set_description("loss:%.2f  accuracy:%.2f" % (np.array(losses)[-4:].mean(), np.array(accuracies)[-4:].mean()))

def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy,
        transform=lambda x: x, target_transform=lambda x: x):
  DenseTensor.training = True
  SparseTensor.training = True
  losses, accuracies = [], []

  randperm = np.random.permutation(range(X_train.shape[0]))
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    # if i > 2:
    #   break
    # samp = np.random.randint(0, X_train.shape[0], size=(BS))
    samp = randperm[i*BS:(i+1)*BS]
    x = DenseTensor(transform(X_train[samp]))
    y = target_transform(Y_train[samp])

    # network
    out = model.forward(x)
    loss = lossfn(out, y)

    optim.zero_grad()
    loss.backward()
    # print('MODEL:', model.parameters())
    optim.step()

    cat = np.argmax(out.cpu().data, axis=-1)
    accuracy = (cat == y).mean()

    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    try:
      nnzs = 0
      gradnnzs = 0
      for weight in model.layers:
        # print("LAYER:", weight)
        nnzs += weight.count_nnzs()
        # gradnnzs += weight.accgrad.count_nnzs()
        t.set_description("loss:%.2f  accuracy:%.2f  nnz:%i" % (np.array(losses)[-4:].mean(), np.array(accuracies)[-4:].mean(), nnzs))
    except Exception as e:
      t.set_description("loss:%.2f  accuracy:%.2f" % (np.array(losses)[-4:].mean(), np.array(accuracies)[-4:].mean()))

def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
  DenseTensor.training = False
  SparseTensor.training = False
  def numpy_eval(Y_test, num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange((len(Y_test)-1)//BS+1, disable=os.getenv('CI') is not None):
      x = DenseTensor(transform(X_test[i*BS:(i+1)*BS]))
      Y_test_preds_out[i*BS:(i+1)*BS] = model.forward(x).cpu().data
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean(), Y_test_preds

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  acc, Y_test_pred = numpy_eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc
