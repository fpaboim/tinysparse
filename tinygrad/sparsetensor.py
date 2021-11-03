# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
import inspect
import functools
import os
from collections import defaultdict
import numpy as np
from .tensor import Device, Tensor
from .densetensor import DenseTensor, GPUBuffer, require_init_gpu, cl_ctx, cl_queue, ane

topk = 1

require_init_gpu()

# **** profiler ****
global cl_ctx, cl_queue, ane

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


DEFAULT_DEVICE = Device.GPU

class SparseTensor(Tensor):
  did_float_warning = False
  training = True
  ops = defaultdict(dict)

  def __init__(self, dense_data=[], data=[], idxs=[], nnzs=[], ellw=None,
               shape=None, randinit=[], randsparsity=0.9, device=DEFAULT_DEVICE, requires_grad=True):
    self.device = device
    # print('rand:', randinit)

    if len(randinit)==0:
      if len(data)==0:
        self.shape = dense_data.shape
      else:
        self.shape = shape
      data, idxs, nnzs, ellw = self.to_ell(dense_data)
      datat, idxst, nnzst, ellwt = self.to_ell(dense_data.T)
    else:
      self.shape = randinit
      data, idxs, nnzs, ellw, datat, idxst, nnzst, ellwt = self.make_random(randinit, sparsity=randsparsity)
      # print('data:', data)
      # datat, idxst, nnzst, ellwt = self.make_random(randinit)

    if len(data)==0:
      data, idxs, nnzs, ellw = self.to_ell(dense_data)
      datat, idxst, nnzst, ellwt = self.to_ell(dense_data.T)
      # data = np.expand_dims(data, 1)
      # idxs = np.expand_dims(idxs, 1)
      # nnzs = np.expand_dims(nnzs, 1)
    # else:
    #   print("HAS DATA:")
    #   print(data, idxs, nnzs, ellw)

    self.data = self._move_data(data, device, np.float32)
    self.idxs = self._move_data(idxs, device, np.uint32)
    self.nnzs = self._move_data(nnzs, device, np.uint32)
    self.ellw = ellw

    self.datat = self._move_data(datat, device, np.float32)
    self.idxst = self._move_data(idxst, device, np.uint32)
    self.nnzst = self._move_data(nnzst, device, np.uint32)
    self.ellwt = ellwt

    self.grad, self.requires_grad = None, requires_grad

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return f"<SparseTensor {self.data!r} with grad {(self.grad if self.grad else None)!r}>"

  def assign(self, x):
    self.data = x.data

  # @property
  # def shape(self):
  #   return self.shape

  @property
  def dtype(self):
    return self.data.dtype

  def to_ell(self, mat, ellwidth=None):
    mat = np.array(mat)
    if not ellwidth:
      maxnnz = 0
      for i in range(mat.shape[0]):
        submat = mat[i]
        newmax = len(submat[submat != 0])
        maxnnz = max(maxnnz, newmax)
      ellwidth = maxnnz*2
    # print("ELLW:", ellwidth)
    all_rows = []
    all_idxs = []
    all_nnzs = []
    for row in range(mat.shape[0]):
        rowdata = []
        colidxs = []
        all_nnzs.append(0)
        for col in range(mat.shape[1]):
            val = mat[row][col]
            if val != 0:
                rowdata.append(val)
                colidxs.append(col)
                all_nnzs[-1] += 1
        rowdata = np.array(rowdata)
        rowdata.resize(ellwidth)
        all_rows.append(rowdata)
        colidxs = np.array(colidxs)
        colidxs.resize(ellwidth)
        all_idxs.append(colidxs)
    all_rows = np.array(all_rows).astype(np.float32).flatten()
    all_idxs = np.array(all_idxs).astype(np.uint32).flatten()
    all_nnzs = np.array(all_nnzs).astype(np.uint32)
    return all_rows, all_idxs, all_nnzs, ellwidth

  def make_random(self, shape, sparsity=0.7):
    all_rows = []
    all_idxs = []
    all_nnzs = []
    nnzs = int(shape[1]*(1-sparsity))
    ellwidth = int((nnzs/2)+1)*4
    ellwidth = min(ellwidth, shape[1])
    cols = {}
    for row in range(shape[0]):
      rowdata = np.random.rand(nnzs) / 4#/ (nnzs)
      rowidx = np.random.permutation(shape[1])[:nnzs]
      i = 0
      for col in rowidx:
        if not col in cols.keys():
          cols[col] = [(rowdata[i],row)]
        else:
          cols[col].append((rowdata[i],row))
        i += 1

      while len(rowdata) < ellwidth:
        rowdata = np.concatenate([rowdata, np.array([0])])
        rowidx  = np.concatenate([rowidx, np.array([0])])
      all_rows.append(rowdata)
      all_idxs.append(rowidx)
      all_nnzs.append(nnzs)

    all_rowst = []
    all_idxst = []
    all_nnzst = []
    maxw = 0
    for row in range(shape[1]):
      try:
        all_rowst.append([val[0] for val in cols[row]])
        all_idxst.append([val[1] for val in cols[row]])
        all_nnzst.append(len(cols[row]))
        maxw = max(len(cols[row]), maxw)
        # print('masw:', maxw)
      except:
        all_rowst.append([])
        all_idxst.append([])
        all_nnzst.append(0)
    ellwidtht = (int(maxw/2)+1)*2
    ellwidtht = min(ellwidtht, shape[0])
    # print('ellwt:', ellwidtht)
    for irow in range(len(all_rowst)):
      # print(all_rowst[irow])
      all_rowst[irow] = np.concatenate([all_rowst[irow], np.zeros(ellwidtht-len(all_rowst[irow]))])
      all_idxst[irow] = np.concatenate([all_idxst[irow], np.zeros(ellwidtht-len(all_idxst[irow]))])
      # print(all_rowst[irow])


    all_rows = np.array(all_rows).astype(np.float32).flatten()
    all_idxs = np.array(all_idxs).astype(np.uint32).flatten()
    all_nnzs = np.array(all_nnzs).astype(np.uint32)
    all_rowst = np.array(all_rowst).astype(np.float32).flatten()
    all_idxst = np.array(all_idxst).astype(np.uint32).flatten()
    all_nnzst = np.array(all_nnzst).astype(np.uint32)


    return all_rows, all_idxs, all_nnzs, ellwidth, all_rowst, all_idxst, all_nnzst, ellwidtht

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
    return cls(randinit=shape, **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs):
    return cls(np.eye(dim).astype(np.float32), **kwargs)

  # ***** tinygrad supports CPU and GPU *****
  @staticmethod
  def _move_data(data, device, dtype=np.float32):
    if isinstance(data, GPUBuffer):
      if device == Device.GPU: return data
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

    if data.dtype != dtype and not SparseTensor.did_float_warning:
      # warning? float64 is actually needed for numerical jacobian
      print(f"warning, {data.shape!r} isn't {dtype}, it's {data.dtype}")
      SparseTensor.did_float_warning = True

    if device == Device.GPU:
      require_init_gpu()
      with ProfileOp("toGPU", [data]):
        return GPUBuffer(data.shape, data, dtype)

    elif device == Device.ANE:
      require_init_ane()
      with ProfileOp("toANE", [data]):
        ndata = ane.tensor(data.shape)
        ndata.data()[:] = data
        return ndata
    return data

  def is_sparse(self):
    return True

  def updategrad(self, grad, lr):
    # Weight update
    ctx = self._ctx
    bs = grad[1].shape[0]

    # print("GRAD:", )
    # resa = np.ones(topk*topk).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, grad[0].cl)
    # print('RESA:', resa)

    # print('ctx:', ctx, self.data)
    addvals = cl.Program(ctx.cl_ctx,
     """
    // Every global_id_0 works on a row
    __kernel void addvals(__global  float* matData,     // INPUT MATRIX DATA
                         __global  uint*  colIdx,
                         __global  uint*  rowNnz,
                         float lr,
                         uint   ellwidth,
                         __global  float* updatevals,    // INPUT
                         __global  uint* updatexidx,
                         __global  uint* updateyidx
                         ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint gid2 = get_global_id(1);
      uint topk = get_global_size(0);
      uint bs = get_global_size(1);
      uint baseupdateidx = topk*topk*gid2;
      uint baseidxidx = topk*gid2;
      uint col = updateyidx[baseidxidx+gid];

      for (uint i=0; i<topk; i++) {
        float val = updatevals[baseupdateidx+gid*topk+i];
        uint row = updatexidx[baseidxidx+i];
        for (uint i=0; i<rowNnz[row]; i++) {
          uint idx = row*ellwidth+i;
          if (colIdx[idx] >= col) {
            //printf("\\nFOUND:%i/%i  - idx:%i", colIdx[idx], col, idx);
            if (colIdx[idx] == col) {
              matData[idx] += -val*lr;
              //printf("\\nUPDATE[%i,%i]: %f", row,col, val);
              break;
            } else {
              // insert new column
              //printf("\\nINSERT[%i,%i]: %.2f", row,col, val);
              for (uint j=rowNnz[row]+1; j>i; j--) {
                uint idx2 = row*ellwidth+j;
                matData[idx2] = matData[idx2-1];
                colIdx[idx2] = colIdx[idx2-1];
              }
              matData[idx] = -val*lr;
              colIdx[idx] = col;
              rowNnz[row] += 1;
              break;
            }
          }
        }
        if (rowNnz[row] >= ellwidth) {
          break;
        }
      }
    }""").build().__getattr__('addvals')

    # resa = np.ones(self.shape[0]).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, grad.cl)
    # print('RESA2:', resa)

    # (isize,msize) x (isize,osize) = (msize,osize)
    # print('grad:', grad)
    addvals(ctx.cl_queue, [topk,bs], None,
      self.data.cl, self.idxs.cl, self.nnzs.cl, np.float32(lr), np.uint32(self.ellw), grad[0].cl, grad[1].cl, grad[2].cl)

    addvals2 = cl.Program(ctx.cl_ctx,
     """
    // Every global_id_0 works on a row
    __kernel void addvals2(__global  float* matData,     // INPUT MATRIX DATA
                         __global  uint*  colIdx,
                         __global  uint*  rowNnz,
                         float lr,
                         uint   ellwidth,
                         __global  float* updatevals,    // INPUT
                         __global  uint* updatexidx,
                         __global  uint* updateyidx
                         ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint gid2 = get_global_id(1);
      uint topk = get_global_size(0);
      uint bs = get_global_size(1);
      uint baseupdateidx = topk*topk*gid2;
      uint baseidxidx = topk*gid2;
      uint row = updateyidx[baseidxidx+gid];

      for (uint i=0; i<topk; i++) {
        float val = updatevals[baseupdateidx+gid*topk+i];
        uint col = updatexidx[baseidxidx+i];
        for (uint i=0; i<rowNnz[row]; i++) {
          uint idx = row*ellwidth+i;
          if (colIdx[idx] >= col) {
            //printf("\\nFOUND:%i/%i  - idx:%i", colIdx[idx], col, idx);
            if (colIdx[idx] == col) {
              matData[idx] += -val*lr;
              //printf("\\nUPDATE[%i,%i]: %f", row,col, val);
              break;
            } else {
              // insert new column
              //printf("\\nINSERT[%i,%i]: %.2f", row,col, val);
              for (uint j=rowNnz[row]+1; j>i; j--) {
                uint idx2 = row*ellwidth+j;
                matData[idx2] = matData[idx2-1];
                colIdx[idx2] = colIdx[idx2-1];
              }
              matData[idx] = -val*lr;
              colIdx[idx] = col;
              rowNnz[row] += 1;
              break;
            }
          }
        }
        if (rowNnz[row] >= ellwidth) {
          break;
        }
      }
    }""").build().__getattr__('addvals2')

    # (isize,msize) x (isize,osize) = (msize,osize)
    # print('grad:', grad)
    addvals2(ctx.cl_queue, [topk,bs], None,
      self.datat.cl, self.idxst.cl, self.nnzst.cl, np.float32(lr), np.uint32(self.ellwt), grad[0].cl, grad[1].cl, grad[2].cl)
    self._ctx = None

  def to_(self, device):
    self.data, self.device = self._move_data(self.data, device), device
    if self.grad: self.grad.to_(device)

  def to(self, device):
    ret = Tensor(self.data, device)
    # if self.grad: ret.grad = self.grad.to(device)
    return ret

  def detach(self):
    return Tensor(self.data, device=self.device)

  def backward(self):
    # print('dense shape grad;', self.shape)
    # assert self.shape == (1,)

    # fill in the first grad with one
    # this is "implicit gradient creation"
    self.grad = DenseTensor(np.ones(self.shape, dtype=self.dtype), device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      assert (t0.grad is not None)
      with ProfileOp(t0._ctx.__class__.__name__, [t0.grad], backward=True) as po:
        # print('t0:', t0)
        grads = t0._ctx.backward(t0._ctx, t0.grad.data)
      if len(t0._ctx.parents) == 1:
        grads = [grads]
      # print("PRT:", t0._ctx.parents)
      # print("GRDS:", grads)
      for t, g in zip(t0._ctx.parents, grads):
        # print("T/g:",t,g)
        try:
          if t.is_sparse():
            # print("SPARSE!",t)
            t._ctx = t0._ctx
            gt = g#DenseTensor(g, device=self.device, requires_grad=False)
            t.grad = gt if t.grad is None else (t.grad + gt)
            continue
        except Exception as e:
          print("ERR:", e)
          pass
        if g is not None:
          assert g.shape == t.shape, \
            f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          gt = DenseTensor(g, device=self.device, requires_grad=False)
          t.grad = gt if t.grad is None else (t.grad + gt)


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
    if Tensor.training:
      _mask = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
      return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))
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

class SparseFunction:
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
      po.output = ret = DenseTensor(self.forward(ctx, *[t for t in x], **kwargs),
                   device=ctx.device, requires_grad=any([t.requires_grad for t in x]))
    if ret.requires_grad:
      ret._ctx = ctx
    return ret

def register_sparse(name, fxn, device=Device.CPU):
  SparseTensor.ops[device][name] = fxn
  def dispatch(*x, **kwargs):
    tt = [arg for arg in x if isinstance(arg, SparseTensor)][0]
    x = [DenseTensor(np.array([arg], dtype=tt.dtype), device=tt.device, requires_grad=False) if not (isinstance(arg, DenseTensor) or isinstance(arg, SparseTensor)) else arg for arg in x]
    # print('x:',x)
    f = SparseTensor.ops[tt.device][name]
    # print('CTX:', cl_ctx)
    f.cl_ctx, f.cl_queue, f.ane, f.device = cl_ctx, cl_queue, ane, tt.device
    # print('APPLY:', *x, f)
    return f.apply(f, *x, **kwargs)
  setattr(SparseTensor, name, dispatch)
  if name in ['add', 'sub', 'mul', 'pow', 'matmul']:
    setattr(SparseTensor, f"__{name}__", dispatch)
    setattr(SparseTensor, f"__i{name}__", lambda self,x: self.assign(dispatch(self,x)))
    setattr(SparseTensor, f"__r{name}__", lambda self,x: dispatch(x,self))

for device in [device for device in Device.__dict__.keys() if device[0] != "_"]:
  setattr(SparseTensor, f"{device.lower()}", functools.partialmethod(SparseTensor.to, Device.__dict__[device]))
  setattr(SparseTensor, f"{device.lower()}_", functools.partialmethod(SparseTensor.to_, Device.__dict__[device]))

# this registers all the operations
def _register_ops(namespace, device=Device.CPU):
  for name, cls in inspect.getmembers(namespace, inspect.isclass):
    if name[0] != "_":  register_sparse(name.lower(), cls, device=device)

# from tinygrad import ops_cpu
# _register_ops(ops_cpu)
if os.getenv("CHERRY", None) is not None:
  from extra import ops_cherry
  _register_ops(ops_cherry)
try:
  import pyopencl as cl
  # TODO: move this import to require_init_gpu?
  from tinygrad import ops_gpusparse
  _register_ops(ops_gpusparse, device=Device.GPU)
  GPU = True
except ImportError:
  # no GPU support
  GPU = False
ANE = False
