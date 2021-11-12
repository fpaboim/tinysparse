# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
import inspect
import functools
import os
from collections import defaultdict
import numpy as np
from .tensor import Device, Tensor

# **** profiler ****

DEBUG = os.getenv("DEBUG", None) is not None
if DEBUG:
  import atexit, time
  debug_counts, debug_times = defaultdict(int), defaultdict(float)
  def print_debug_exit():
    for name, _ in sorted(debug_times.items(), key=lambda x: -x[1]):
      print(f"{name:>20} : {debug_counts[name]:>6} {debug_times[name]:>10.2f} ms")
  atexit.register(print_debug_exit)

class ProfileOp:
  def __init__(self, name, x, backward=False):
    self.name, self.x, self.output = f"back_{name}" if backward else name, x, None
  def __enter__(self):
    if DEBUG: self.st = time.time()
    return self
  def __exit__(self, *junk):
    if DEBUG:
      if cl_queue is not None:
        cl_queue.finish()
      et = (time.time()-self.st)*1000.
      debug_counts[self.name] += 1
      debug_times[self.name] += et
      print(f"{self.name:>20} : {et:>7.2f} ms {str([y.shape for y in self.x]):>40} {'-> '+str(self.output.shape) if self.output is not None else ''}")

# **** GPU functions ****

cl_ctx, cl_queue = None, None
def require_init_gpu():
  global cl_ctx, cl_queue
  if not GPU: raise Exception("No GPU Support, install pyopencl")
  if cl_queue is None:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
      devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
      print("DEVICE:" 'CPU')
    else:
      print("DEVICE:" 'GPU')
    cl_ctx = cl.Context(devices=devices)
    # this is an in-order command queue
    cl_queue = cl.CommandQueue(cl_ctx)

class GPUBuffer:
  def __init__(self, shape, hostbuf=None, dtype=np.float32):
    self.shape, self.dtype = tuple(shape), dtype
    self.cl = hostbuf.cl if isinstance(hostbuf, GPUBuffer) else \
      cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape),
                hostbuf=hostbuf.astype(dtype).ravel() if hostbuf is not None else None)

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

# **** ANE functions ****

ane = None
def require_init_ane():
  global ane
  if ane is None:
    import accel.ane.lib.ane as anelib, tinygrad.ops_ane
    ane = anelib.ANE()

# **** start with two base classes, Tensor and Function ****

DEFAULT_DEVICE = Device.GPU

class DenseTensor(Tensor):
  did_float_warning = False
  training = True
  ops = defaultdict(dict)

  def __init__(self, data, device=DEFAULT_DEVICE, requires_grad=True):
    self.device, self.data = device, self._move_data(data, device, np.float32)

    self.grad, self.requires_grad = None, requires_grad

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return f"<DenseTensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"

  def assign(self, x):
    self.data = x.data

  @staticmethod
  def _move_data(data, device, dtype=np.float32):
    if isinstance(data, GPUBuffer):
      if device == Device.GPU:
        return data
      old = data
      data = np.empty(old.shape, dtype=dtype)
      with ProfileOp("toCPU", [data]):
        cl.enqueue_copy(cl_queue, data, old.cl, is_blocking=True)

    elif "ANETensor" in str(type(data)):
      if device == Device.ANE: return data
      with ProfileOp("toCPU", [data]):
        data = data.data().astype(dtype)

    if not isinstance(data, np.ndarray):
      data = np.array(data, dtype=dtype)

    if data.dtype != dtype and not DenseTensor.did_float_warning:
      # warning? float64 is actually needed for numerical jacobian
      print(f"warning, {data.shape!r} isn't float32, it's {data.dtype}")
      DenseTensor.did_float_warning = True

    if device == Device.GPU:
      require_init_gpu()
      with ProfileOp("toGPU", [data]):
        return GPUBuffer(data.shape, data)

    elif device == Device.ANE:
      require_init_ane()
      with ProfileOp("toANE", [data]):
        ndata = ane.tensor(data.shape)
        ndata.data()[:] = data
        return ndata
    return data

  def to_(self, device):
    self.data, self.device = self._move_data(self.data, device), device
    if self.grad: self.grad.to_(device)

  def to(self, device):
    ret = DenseTensor(self.data, device)
    if self.grad: ret.grad = self.grad.to(device)
    return ret

  def backward(self):
    # print('dense shape grad;', self.shape)
    # assert self.shape == (1,)

    # fill in the first grad with one
    # this is "implicit gradient creation"
    self.grad = DenseTensor(np.ones(self.shape, dtype=self.dtype), device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      assert (t0.grad is not None)
      with ProfileOp(t0._ctx.__class__.__name__, [t0.grad], backward=True) as po:
        # print('t0:', t0, t0.grad.cpu().data)
        grads = t0._ctx.backward(t0._ctx, t0.grad.data)
      if len(t0._ctx.parents) == 1:
        grads = [grads]
      # print("PRT:", t0._ctx.parents)
      # print("GRDS:", grads)
      for t, g in zip(t0._ctx.parents, grads):
        # print("T/g:",t,g)
        #     # print("SPARSE!",g)
        #     gt = g#DenseTensor(g, device=self.device, requires_grad=False)
        #     t.grad = gt if t.grad is None else (t.grad + gt)
        #     continue
        # except Exception as e:
        #   print("ERR:", e)
        #   pass
        if g is not None:
          print('T/G:', t,g,t.shape, g.shape)
          # if not (not isinstance(t, DenseTensor)) and (not isinstance(t,GPUBuffer)):
          assert g.shape == t.shape, \
            f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          if isinstance(g, DenseTensor):
            # print("DENSETENSOR")
            gt = g
          else:
            gt = DenseTensor(g, device=self.device, requires_grad=False)
          t.grad = gt if t.grad is None else (t.grad + gt)
          if t.is_sparse():
            print('T:', t)
          # try:
          #   if t.is_sparse():
          #     t._ctx = t0._ctx
          # except:
          #   pass
          # print("SET GRAD:", t, gt)

  def detach(self):
    return DenseTensor(self.data, device=self.device)

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  # ***** creation helper functions *****

  @classmethod
  def zeros(cls, *shape, **kwargs):
    return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs):
    return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs):
    return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

  @classmethod
  def arange(cls, stop, start=0, **kwargs):
    return cls(np.arange(start=start, stop=stop).astype(np.float32), **kwargs)

  @classmethod
  def uniform(cls, *shape, **kwargs):
    return cls((np.random.uniform(-1., 1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs):
    return cls(np.eye(dim).astype(np.float32), **kwargs)

  # ***** tinygrad supports CPU and GPU *****

  # ***** non first class ops *****

  def __getitem__(self, val):
    arg = []
    if val is not None:
      for i, s in enumerate(val if isinstance(val, (list, tuple)) else [val]):
        if isinstance(s, int):
          arg.append((s, s + 1))
        else:
          arg.append((s.start if s.start is not None else 0,
            (s.stop if s.stop >=0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
          assert s.step is None or s.step == 1
    return self.slice(arg = arg + [(0,self.shape[i]) for i in range(len(arg), len(self.shape))])

  def pad2d(self, padding):
    return self[:, :, -padding[2]:self.shape[2]+padding[3], -padding[0]:self.shape[3]+padding[1]]

  def dot(self, w):
    if w.is_sparse():
      # xt = self.transpose()
      # wt = w#.transpose()
      # print("SPRSE MATMUL", wt, xt)
      x = w.matmul(self)
      # print("Xt:", x)
      return x
    return self.matmul(w)

  def mean(self, axis=None):
    out = self.sum(axis=axis)
    return out * (np.prod(out.shape)/np.prod(self.shape))

  def sqrt(self):
    return self.pow(0.5)

  def div(self, y):
    return self * (y ** -1.0)
  __truediv__ = div

  def sigmoid(self):
    e = self.exp()
    return e.div(1 + e)

  def swish(self):
    return self * self.sigmoid()

  def relu6(self):
    return self.relu() - (self-6).relu()

  def hardswish(self):
    return self * (self+3).relu6() * (1/6)

  def tanh(self):
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def leakyrelu(self, neg_slope=0.01):
    return self.relu() - (-neg_slope*self).relu()

  def softmax(self):
    ns = list(self.shape)[:-1]+[1]
    m = self.max(axis=len(self.shape)-1).reshape(shape=ns)
    e = (self - m).exp()
    ss = e.sum(axis=len(self.shape)-1).reshape(shape=ns)
    return e.div(ss)

  def logsoftmax(self):
    ns = list(self.shape)[:-1]+[1]
    m = self.max(axis=len(self.shape)-1).reshape(shape=ns)
    ss = m + (self-m).exp().sum(axis=len(self.shape)-1).reshape(shape=ns).log()
    return self - ss

  def dropout(self, p=0.5):
    if DenseTensor.training:
      _mask = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
      return self * DenseTensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))
    else:
      return self

  def softplus(self, limit=20, beta=1):
    # safe softplus - 1/beta*log(1 + exp(beta*x)) (PyTorch)
    eb = (self*beta).exp()
    ret = (1 + eb).log()
    return (1/beta)*ret

  def mish(self):
    return self * (self.softplus().tanh()) # x*tanh(softplus(x))

  def abs(self):
    return self.relu() + (-1.0*self).relu()

  def sign(self):
    return self / (self.abs() + 1e-10)

  def _pool2d(self, py, px):
    xup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px]
    return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//py, py, xup.shape[3]//px, px))

  def avg_pool2d(self, kernel_size=(2,2)):
    return self._pool2d(*kernel_size).mean(axis=(3,5))

  def max_pool2d(self, kernel_size=(2,2)):
    return self._pool2d(*kernel_size).max(axis=(3,5))

# An instantiation of the Function is the Context
class Function:
  def __new__(cls, *args, **kwargs):
    cls.forward = staticmethod(cls.forward)
    cls.backward = staticmethod(cls.backward)
    return super().__new__(cls)

  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    # print("APPLY:", x, kwargs)
    ctx = self(*x) # self - operation i.e 'add', 'sub', etc.
    # use default params
    params = inspect.signature(self.forward).parameters
    for p in params.values():
      if p.default is not p.empty:
        setattr(ctx, p.name, p.default)
    # overwrite with passed params
    for k, v in kwargs.items():
      setattr(ctx, k, v)
    with ProfileOp(ctx.__class__.__name__, x) as po:
      res = self.forward(ctx, *[t.data if 'DenseTensor' in t.__class__.__name__ else t for t in x], **kwargs)
      # print("RES:", res)
      po.output = ret = DenseTensor(res,
                   device=ctx.device, requires_grad=any([t.requires_grad for t in x]))
    if ret.requires_grad:
      ret._ctx = ctx
    return ret

def register(name, fxn, device=Device.CPU):
  DenseTensor.ops[device][name] = fxn
  def dispatch(*x, **kwargs):
    #print('X:',x)
    tt = [arg for arg in x if isinstance(arg, DenseTensor)][0]
    # x = [DenseTensor(np.array([arg], dtype=tt.dtype), device=tt.device, requires_grad=False) if not isinstance(arg, DenseTensor) else arg for arg in x]
    xout = []
    for arg in x:
      if not isinstance(arg, DenseTensor):
        if isinstance(arg, GPUBuffer):
          xout.append(DenseTensor(arg, device=tt.device, requires_grad=False))
        else:
          if not 'Tensor' in arg.__class__.__name__:
            xout.append(DenseTensor(np.array([arg], dtype=tt.dtype), device=tt.device, requires_grad=False))
          else:
            xout.append(arg)
      else:
        xout.append(arg)
    x = xout
    #print('X2:',x,tt)
    f = DenseTensor.ops[tt.device][name]
    f.cl_ctx, f.cl_queue, f.ane, f.device = cl_ctx, cl_queue, ane, tt.device
    return f.apply(f, *x, **kwargs)
  setattr(DenseTensor, name, dispatch)
  if name in ['add', 'sub', 'mul', 'pow', 'matmul']:
    setattr(DenseTensor, f"__{name}__", dispatch)
    setattr(DenseTensor, f"__i{name}__", lambda self,x: self.assign(dispatch(self,x)))
    setattr(DenseTensor, f"__r{name}__", lambda self,x: dispatch(x,self))

for device in [device for device in Device.__dict__.keys() if device[0] != "_"]:
  setattr(DenseTensor, f"{device.lower()}", functools.partialmethod(DenseTensor.to, Device.__dict__[device]))
  setattr(DenseTensor, f"{device.lower()}_", functools.partialmethod(DenseTensor.to_, Device.__dict__[device]))

# this registers all the operations
def _register_ops(namespace, device=Device.CPU):
  for name, cls in inspect.getmembers(namespace, inspect.isclass):
    if name[0] != "_":  register(name.lower(), cls, device=device)

from tinygrad import ops_cpu
_register_ops(ops_cpu)
if os.getenv("CHERRY", None) is not None:
  from extra import ops_cherry
  _register_ops(ops_cherry)
try:
  # print("TRY GPU")
  import pyopencl as cl
  # TODO: move this import to require_init_gpu?
  from tinygrad import ops_gpu
  _register_ops(ops_gpu, device=Device.GPU)
  GPU = True
except ImportError:
  # no GPU support
  GPU = False
  print("NO GPU")
ANE = False

require_init_gpu()
