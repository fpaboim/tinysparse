# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
import inspect
import functools
import os
from collections import defaultdict
import numpy as np
from .tensor import Device, Tensor
from .densetensor import DenseTensor, GPUBuffer, require_init_gpu, cl_ctx, cl_queue, ane

topk = 2

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

  def __init__(self, dense_data=[], from_datas={}, idxs=[], nnzs=[], ellw=None,
               shape=None, randinit=[], randsparsity=0.01, bs=32, device=DEFAULT_DEVICE, requires_grad=True, ctx=None):
    self.device = device

    if len(randinit)==0:
      if len(from_datas.keys())==0:
        self.shape = dense_data.shape
        data, idxs, nnzs, ellw = self.to_ell(dense_data)
        datat, idxst, nnzst, ellwt = self.to_ell(dense_data.T)
      else:
        assert(shape!=None)
        self.shape = shape
    else:
      self.shape = randinit
      data, idxs, nnzs, ellw, datat, idxst, nnzst, ellwt = self.make_random(randinit, sparsity=randsparsity)
      # print('data:', data)
      # datat, idxst, nnzst, ellwt = self.make_random(randinit)


    if len(from_datas.keys()) > 0:
      self.data = self._move_data(from_datas['data'], device, np.float32)
      self.idxs = self._move_data(from_datas['idxs'], device, np.uint32)
      self.nnzs = self._move_data(from_datas['nnzs'], device, np.uint32)
      self.ellw = from_datas['ellw']

      self.datat = self._move_data(from_datas['datat'], device, np.float32)
      self.idxst = self._move_data(from_datas['idxst'], device, np.uint32)
      self.nnzst = self._move_data(from_datas['nnzst'], device, np.uint32)
      self.ellwt = from_datas['ellwt']
    else:
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
    self._ctx = ctx

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
      ellwidth = maxnnz

    # print("ELLW:", ellwidth)
    ellwidth = min(int(np.sqrt(ellwidth)+1)**2, mat.shape[1])
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

    # while (not all_rows[:,-1].any()):
    #     all_rows = all_rows[:,:-1]
    #     all_idxs = all_idxs[:,:-1]
    #     ellwidth -= 1

    all_rows = np.array(all_rows).astype(np.float32).flatten()
    all_idxs = np.array(all_idxs).astype(np.uint32).flatten()
    all_nnzs = np.array(all_nnzs).astype(np.uint32)

    return all_rows, all_idxs, all_nnzs, ellwidth

  def make_random(self, shape, sparsity=0.7):
    all_rows = []
    all_idxs = []
    all_nnzs = []
    nnzs = min(int(shape[1]*(1-sparsity))+1, shape[1])
    ellwidth = int(nnzs**.5+1)**2
    ellwidth = min(shape[1], ellwidth)
    ellwidth = max(ellwidth, shape[1])
    # print('ellw:', ellwidth, shape)
    cols = {}
    for row in range(shape[0]):
      rowdata = np.random.rand(nnzs) / 100
      rowidx = sorted(np.random.permutation(shape[1])[:nnzs])
      i = 0

      for col in rowidx:
        val = rowdata[i]
        if not col in cols.keys():
          cols[col] = [(val,row)]
        else:
          cols[col].append((val,row))
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
    # ellwidtht = (int(maxw/2)+1)*2
    ellwidtht = shape[0]
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
    # print("MOVE DATA", data)
    if isinstance(data, GPUBuffer):
      if device == Device.GPU: return data
      old = data
      data = np.empty(old.shape, dtype=dtype)
      with ProfileOp("toCPU", [data]):
        cl.enqueue_copy(cl_queue, data, old.cl, is_blocking=True)
      return data

    elif "ANETensor" in str(type(data)):
      if device == Device.ANE: return data
      with ProfileOp("toCPU", [data]):
        data = data.data().astype(dtype)

    # print("DATA", data)
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

  def get_nnzs(self):
    global cl_ctx, cl_queue
    ctx = cl_ctx

    dim = (self.shape[0])
    res= np.zeros(dim).astype(np.uint32)
    cl.enqueue_copy(cl_queue, res, self.nnzs.cl)
    # print('RES:', res.sum())
    return res

  def count_nnzs(self):
    global cl_ctx, cl_queue
    ctx = cl_ctx

    dim = (self.shape[0])
    res= np.zeros(dim).astype(np.uint32)
    cl.enqueue_copy(cl_queue, res, self.nnzs.cl, is_blocking=True)
    # print('RES:', res.sum())
    return res.sum()

  def to_numpy(self, dual=False):
    global cl_ctx, cl_queue
    ctx = cl_ctx

    if dual:
      data, cols, nnzs, ellw, shape = self.datat, self.idxst, self.nnzst, self.ellwt, np.array(self.shape).T
    else:
      data, cols, nnzs, ellw, shape = self.data, self.idxs, self.nnzs, self.ellw, self.shape

    dim = shape[0]*ellw
    newdata= np.zeros(dim).astype(np.float32)
    cl.enqueue_copy(cl_queue, newdata, data.cl, is_blocking=True)

    newcols= np.zeros(dim).astype(np.uint32)
    cl.enqueue_copy(cl_queue, newcols, cols.cl, is_blocking=True)

    newnnzs= np.zeros(shape[0]).astype(np.uint32)
    cl.enqueue_copy(cl_queue, newnnzs, nnzs.cl, is_blocking=True)

    out = np.zeros(shape)
    for row in range(shape[0]):
        # print('nnzs:', newnnzs[row])
        for icol in range(int(newnnzs[row])):
          # print('newcol:', newcols[row*ellw+icol])
          out[row,newcols[row*ellw+icol]] = newdata[row*ellw+icol]
    return out

  def updategrad(self, grad, lr):
    # Weight update
    # print("UPDATE GRAD", grad)
    global cl_ctx, cl_queue
    ctx = cl_ctx

    adddense = cl.Program(ctx,"""
    // Every global_id_0 works on a row
    __kernel void adddense(__global  float* matData,     // INPUT MATRIX DATA
                            __global  uint*  colIdx,
                            __global  uint*  rowNnz,
                            float  lr,
                            uint   ellwidth,
                            __global  float* matDataAdd,     // INPUT MATRIX DATA
                            __global  uint*  colIdxAdd,
                            __global  uint*  rowNnzAdd,
                            uint ellwidthAdd
                            ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint nrows = get_global_size(0);

      uint nnz    = rowNnz[gid];

      uint baseidxs = gid*ellwidth;
      uint baseidxd = gid*ellwidthAdd;

      uint nnzadd = rowNnzAdd[gid];
      //printf("\\nNNZs: %i   GID:%i", nnzadd, gid);

      for (uint i=0; i<nnzadd; i++) {
        float addval = matDataAdd[baseidxd+i] * lr;
        uint addcol = colIdxAdd[baseidxd+i];

        uint refcol = colIdx[baseidxs+i];
        uint m = 0;
        while (addcol > refcol) {
          m += 1;
          refcol = colIdx[baseidxs+i+m];
        }

        //printf("\\nADD VAL:%.2f  ADDCOL:%i  idxs/d:(%i/%i)  gid/i:(%i/%i)", addval, addcol, baseidxs, baseidxd, gid,i);
        if (addval == 0.0) {
          //printf("\\nZERO VAL, CONT: %.2f - %i", addval, gid);
          continue;
        }
        if (addcol == refcol) {
          matData[baseidxs+i+m] += addval;
          //printf("\\nINCREMENT: %.2f",addval);
        } else {
          if (rowNnz[gid] >= ellwidth) {
            break;
          }
          if (addcol > refcol) {
            rowNnz[gid] += 1;
            //printf("\\nSET VAL0:%.2f idx:%i/%i  col:%i", addval, baseidxs+i, baseidxd+i, colIdx[i]);
            matData[baseidxs+i+m] = addval;
            colIdx[baseidxs+i+m] = addcol;
            continue;
          }
          for (uint j=nnz; j>i+m; j--) {
            //printf("\\nMOVE:%.2f", matData[baseidx+j-1]);
            colIdx[baseidxs+j] = colIdx[baseidxs+j-1];
            matData[baseidxs+j] = matData[baseidxs+j-1];
          }
          rowNnz[gid] += 1;
          nnz = rowNnz[gid];

          //printf("\\nSET VAL:%.2f idx:%i/%i  col:%i", addval, baseidxs+i, baseidxd+i, colIdx[i]);
          matData[baseidxs+i+m] = addval;
          colIdx[baseidxs+i+m] = addcol;
          if (nnz >= ellwidth)
            break;
        }
      }
    }""").build().__getattr__('adddense')

    # (isize,msize) x (isize,osize) = (msize,osize)
    # print('grad:', grad)
    adddense(cl_queue, [grad.shape[0]], None,
      self.data.cl, self.idxs.cl, self.nnzs.cl, np.float32(lr), np.uint32(self.ellw),
      grad.datat.cl, grad.idxst.cl, grad.nnzst.cl, np.uint32(topk))


    # (isize,msize) x (isize,osize) = (msize,osize)
    # print('grad:', grad)
    adddense(cl_queue, [grad.shape[1]], None,
      self.datat.cl, self.idxst.cl, self.nnzst.cl, np.float32(lr), np.uint32(self.ellwt),
      grad.data.cl, grad.idxs.cl, grad.nnzs.cl, np.uint32(topk))
    # self._ctx = None

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
    self.grad = densetensor(np.ones(self.shape, dtype=self.dtype), device=self.device, requires_grad=false)

    for t0 in reversed(self.deepwalk()):
      # print("t0:",t0)
      assert (t0.grad is not none)
      with profileop(t0._ctx.__class__.__name__, [t0.grad], backward=true) as po:
        # print('t0:', t0, t0.grad.cpu().data)
        grads = t0._ctx.backward(t0._ctx, t0)
      if len(t0._ctx.parents) == 1:
        grads = [grads]
      # print("prt:", t0._ctx.parents)
      # print("grds:", grads)
      for t, g in zip(t0._ctx.parents, grads):
        # print("t/g:",t,g)
        # try:
        #   if t.is_sparse():
        #     # print("sparse!",g)
        #     # t._ctx = t0._ctx
        #     gt = g#densetensor(g, device=self.device, requires_grad=false)
        #     t.grad = gt if t is none else (t + gt)
        #     continue
        # except exception as e:
        #   print("err:", e)
        #   pass
        if g is not none:
          assert g.shape == t.shape, \
            f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          gt = DenseTensor(g.data, device=self.device, requires_grad=false)
          t.grad = gt if t is none else (t + gt)
          print("SET GRAD:", t, gt)

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
      # is_sparsematmul = ctx.__class__.__name__ == 'Matmul' and isinstance(x[0], SparseTensor)
      res = self.forward(ctx, *[t.data if 'DenseTensor' in t.__class__.__name__ else t for t in x], **kwargs)
      if isinstance(res, GPUBuffer):
        po.output = ret = DenseTensor(res,
                     device=ctx.device, requires_grad=any([t.requires_grad for t in x]))
      else:
        po.output = ret = res
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
    ret= f.apply(f, *x, **kwargs)
    # print('APPLY:', *x, f, ret)
    return ret
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
