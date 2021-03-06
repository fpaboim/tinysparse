import functools
import pyopencl as cl
import numpy as np
from .sparsetensor import SparseFunction, SparseTensor
from .densetensor import GPUBuffer, DenseTensor

class GradData:
  def __init__(self, data, xidx, yidx, shape):
    self.data = data
    self.xidx = xidx
    self.yidx = yidx
    self.shape = shape

def buffer_new(ctx, shape, zero=False, dtype=np.float32):
  return GPUBuffer(shape, hostbuf=None if not zero else np.zeros(shape, dtype=dtype))

def buffer_np(ctx, x):
  return cl.Buffer(ctx.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

@functools.lru_cache
def clbuild(cl_ctx, name, prg):
  return cl.Program(cl_ctx, prg).build().__getattr__(name)

def uint2(x, y):
  return np.array((x,y), dtype=cl.cltypes.uint2)
i32 = np.int32

# ************* unary ops *************
def unary_op(ctx, code, x):
  ret = buffer_new(ctx, x.shape)
  unop = clbuild(ctx.cl_ctx, "unop", """
  __kernel void unop(__global const float *a_g, __global float *res_g) {
    int gid = get_global_id(0);
    float a = a_g[gid];
    res_g[gid] = """+code+""";
  }""")
  unop(ctx.cl_queue, [np.prod(ret.shape)], None, x.cl, ret.cl)
  return ret

class ReLU(SparseFunction):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'max(a, (float)0.)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'a * (b >= 0)', grad_output, input)

class Log(SparseFunction):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'log(a)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'a / b', grad_output, input)

class Exp(SparseFunction):
  def forward(ctx, input):
    ret = unary_op(ctx, 'exp(a)', input)
    ctx.save_for_backward(ret)
    return ret

  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op(ctx, 'a * b', grad_output, ret)

# ************* reduce ops *************

def reduce_op(ctx, code, code2, inp, axis=None, start="0.0"):
  if axis is None:
    # full reduce
    osize = [1]*len(inp.shape)
  else:
    osize = np.array(inp.shape)
    osize[list(axis)] = 1
  ret = buffer_new(ctx, osize)
  if axis is None:
    ret.shape = (1,)

  # TODO: this is insanely slow
  reduce = clbuild(ctx.cl_ctx, "reduce", """
  __kernel void reduce(__global const float *a_g, int sz, __global float *res_g, int prod, int n_dims,
                       __global const int *shape_x, __global const int *shape_ret) {
    int gid = get_global_id(0);

    float out = """+start+""";
    for (int x = 0; x < sz; x++) {
      int idx = 0;  // compute index into a_g
      int tprod = prod;
      int tsz = sz;
      for (int dim = 0; dim < n_dims; dim++) {
        idx *= shape_x[dim];
        if (shape_x[dim] == shape_ret[dim]) {   // dim from gid, don't reduce
          tprod /= shape_x[dim];
          idx += (gid / tprod) % shape_x[dim];
        } else {  // dim from x
          tsz /= shape_x[dim];
          idx += (x / tsz) % shape_x[dim];
        }
      }
      float a = a_g[idx];
      """+code+""";
    }
    res_g[gid] = """+code2+""";
  }""")
  reduce(ctx.cl_queue, [np.prod(osize)], None, inp.cl,
    i32(np.prod(inp.shape)//np.prod(osize)), ret.cl,
    i32(np.prod(osize)), i32(len(osize)),
    buffer_np(ctx, np.array(inp.shape, dtype=np.int32)),
    buffer_np(ctx, np.array(osize, dtype=np.int32)))
  return ret

class Sum(SparseFunction):
  def forward(ctx, input, axis=None):
    if isinstance(axis, int): axis = [axis]
    ctx.save_for_backward(input, axis)
    ret = reduce_op(ctx, "out += a", "out", input, axis=axis)
    if axis is not None:
      ret.shape = tuple([input.shape[i] for i in range(len(input.shape)) if i not in axis])
    return ret

  def backward(ctx, grad_output):
    input, axis = ctx.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    output = GPUBuffer(shape, hostbuf=grad_output)
    return binary_op(ctx, 'a+b', output, buffer_new(ctx, input.shape, zero=True))

class Max(SparseFunction):
  def forward(ctx, input, axis=None):
    if isinstance(axis, int): axis = [axis]
    ret = reduce_op(ctx, "out = max(a,out)", "out", input, axis=axis, start="-INFINITY")
    ctx.save_for_backward(input, axis, ret)
    if axis is not None:
      ret.shape = tuple([input.shape[i] for i in range(len(input.shape)) if i not in axis])
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    ret2 = binary_op(ctx, "1.0*(a==b)", input, GPUBuffer(shape, ret))
    div = reduce_op(ctx, "out += a", "out+1e-10", ret2, axis=axis)
    ret3 = binary_op(ctx, "a/b", ret2, GPUBuffer(shape, div))
    return binary_op(ctx, 'a*b', ret3, GPUBuffer(shape, grad_output))

# ************* binary ops *************

@functools.lru_cache
def get_binop_prg(cl_ctx, code, complist):
  ndims = len(complist)
  args = "".join([f", int d{i}" for i in range(ndims)] + [f", int p{i}" for i in range(ndims-1)])
  compute_idx_rets = "".join([f"\n    int idx_ret{i} = (gid0 / {f'p{i}' if i < ndims-1 else '1'}) % d{i};" for i in range(ndims)])

  idx_exprs = ["0", "0"] # [idx_x, idx_y]
  for i in range(ndims):
    for j in range(2):
      if complist[i][j]:
        idx_exprs[j] = "idx_ret%d + d%d*(%s)" % (i, i, idx_exprs[j])

  return cl.Program(cl_ctx, """__kernel void binop(__global const float *x_g, __global const float *y_g, __global float *res_g"""+args+""") {
    int gid0 = get_global_id(0);"""+compute_idx_rets+"""
    float a = x_g["""+idx_exprs[0]+"""];
    float b = y_g["""+idx_exprs[1]+"""];
    res_g[gid0] = """+code+""";\n}""").build()

def binary_op(ctx, code, x, y):
  n_dims = max(len(x.shape), len(y.shape))
  shape_x, shape_y = np.ones(n_dims, dtype=np.int32), np.ones(n_dims, dtype=np.int32)
  shape_x[:len(x.shape)] = np.array(x.shape, dtype=np.int32)
  shape_y[:len(y.shape)] = np.array(y.shape, dtype=np.int32)
  if not np.all((shape_x == 1) | (shape_y == 1) | (shape_x == shape_y)):
    raise Exception(f"binary op unbroadcastable shape mismatch: {x.shape} vs {y.shape}")
  shape_ret = np.maximum(shape_x, shape_y)

  dimlist, complist = [], [] # note: len(dimlist) may be less than n_dims
  def push(dim, comp):
    if len(complist) > 0 and complist[-1] == comp:
      dimlist[-1] *= dim
    elif comp != (False, False):
      dimlist.append(dim); complist.append(comp)
  for i in range(n_dims): # group together any adjacent dimensions that we can to simplify broadcasting
    push(i32(max(shape_x[i], shape_y[i])), (shape_x[i] > 1, shape_y[i] > 1))

  prg = get_binop_prg(ctx.cl_ctx, code, tuple(complist))
  ret = buffer_new(ctx, shape_ret, zero=True)
  prod_list = np.array(dimlist, dtype=i32)[-1::-1].cumprod(dtype=i32)[-1::-1] # take cumprod from back to front
  prg.binop(ctx.cl_queue, [prod_list[0]] if len(dimlist) > 0 else [1], None, x.cl, y.cl, ret.cl, *dimlist, *(prod_list[1:]))
  return ret

def unbroadcast(ctx, out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op(ctx, "out += a", "out", out, sum_axis)

class Add(SparseFunction):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op(ctx, 'a+b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_x, shape_x), unbroadcast(ctx, grad_y, shape_y),

class Sub(SparseFunction):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op(ctx, 'a-b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, unary_op(ctx, '-a', grad_output)
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_x, shape_x), unbroadcast(ctx, grad_y, shape_y),

class Mul(SparseFunction):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'a*b', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op(ctx, 'a*b', y, grad_output)
    grad_y = binary_op(ctx, 'a*b', x, grad_output)
    return unbroadcast(ctx, grad_x, x.shape), unbroadcast(ctx, grad_y, y.shape),

class Pow(SparseFunction):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'pow(a,b)', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'b * (pow((float)a, (float)(b-1.0)))', x, y))
    grad_y = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'pow(a, (float)b) * log(a);', x, y))
    return unbroadcast(ctx, grad_x, x.shape), unbroadcast(ctx, grad_y, y.shape),

# ************* movement ops *************

class Reshape(SparseFunction):
  def forward(ctx, x, shape):
    x.data.shape = tuple(shape)
    ctx.save_for_backward(x)
    return x

  def backward(ctx, grad_output):
    in_shape = ctx.saved_tensors
    return in_shape

def perm_axis(ctx, inp, order):
  # print("PERM:", inp, order)
  osize = np.array(inp.shape)[list(order)]
  ret = buffer_new(ctx, osize)
  perm = clbuild(ctx.cl_ctx, "perm", """
  __kernel void perm(__global const float *a_g, __global float *res_g, int n_axis,
                       __global const int *shape, __global const int *order) {
    int gid = get_global_id(0);
    int gi = gid;
    int idx = 0;
    for(int i = n_axis-1; i>-1; i--) {
      int stride = 1;
      for(int j=order[i]+1; j<n_axis; j++) stride *= shape[j];
      idx += (gi % shape[order[i]])*stride;
      gi /= shape[order[i]];
    }
    res_g[gid] = a_g[idx];
    }""")
  perm(ctx.cl_queue, [np.prod(osize)], None, inp.cl, ret.cl, i32(len(osize)),
    buffer_np(ctx, np.array(inp.shape, dtype=np.int32)),
    buffer_np(ctx, np.array(order, dtype=np.int32)))
  # print("RAN")
  return ret

class Transpose(SparseFunction):
  def forward(ctx, x):
    # print("T FWD:", x)
    newdata = {
     'data': x.datat,
     'idxs': x.idxst,
     'nnzs': x.nnzst,
     'ellw': x.ellwt,
     'datat': x.data,
     'idxst': x.idxs,
     'nnzst': x.nnzs,
     'ellwt': x.ellw,
    }
    newshape = tuple(np.array(x.shape).T)
    ret = SparseTensor(from_datas=newdata, shape=newshape)
    return ret

  def backward(ctx, grad_output):
    return perm_axis(ctx, grad_output, np.argsort((1,0)))

# TODO: merge this with perm axis
def inner_slice(ctx, x, arg):
  shift = [y[0] for y in arg]
  oshape = [y[1]-y[0] for y in arg]
  ret = buffer_new(ctx, oshape)
  gslice = clbuild(ctx.cl_ctx, "gslice", """
  __kernel void gslice(__global const float *input, __global float *output, int prod, int n_dims,
                       __global const int *shape_x, __global const int *shape_ret,
                       __global const int *shift) {
    int gid = get_global_id(0);
    int iptr = 0;
    int zero = 1;
    for (int dim = 0; dim < n_dims; dim++) {
      prod /= shape_ret[dim];
      int sidx = (gid / prod) % shape_ret[dim] + shift[dim];
      zero &= (sidx >= 0 && sidx < shape_x[dim]);
      iptr = (iptr * shape_x[dim]) + sidx;
    }
    output[gid] = zero ? input[iptr] : 0.0;
  }""")
  gslice(ctx.cl_queue, [np.prod(ret.shape)], None,
    x.cl, ret.cl, i32(np.prod(ret.shape)), i32(len(ret.shape)),
    buffer_np(ctx, np.array(x.shape, dtype=np.int32)),
    buffer_np(ctx, np.array(ret.shape, dtype=np.int32)),
    buffer_np(ctx, np.array(shift, dtype=np.int32)))
  return ret

class Slice(SparseFunction):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    return inner_slice(ctx, x, arg)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(ctx, grad_output, narg)

# ************* processing ops *************

class Matmul(SparseFunction): # input and weights are swapped, legacy..
  def forward(ctx, weight, input):
    # print("WEIGHT/input:", weight.shape, input.shape)
    # print(input.shape, weight.shape)
    # assert weight.shape[-2] == input.shape[-1]

    # if not weight.m:
    #   weight.m = DenseTensor(np.zeros((input.shape[0], weight.shape[1])))

    isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])
    outshape = np.array([input.shape[-2], weight.shape[-1]])
    # print("OUT:", outshape, isize, msize, osize)
    outdata = np.zeros(outshape)
    ret = DenseTensor(outdata)
    # ret = buffer_new(ctx.cl_ctx, outshape, zero=True)
    # print("RET:", ret)
    # print("RET:", input)

    matmul = clbuild(ctx.cl_ctx, "matmul", """
    // DENSE x SPARSE
    __kernel void matmul(__global  float* matData,     // INPUT MATRIX DATA
                            __global  uint*  colIdx,
                            __global  uint*  rowNnz,
                            uint   ellwidth,
                            uint   mwidth,
                            uint   ncols,
                            __global  float* vector_x,    // INPUT
                            __global  float* vector_y    // OUTPUT
                            ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint nrows = get_global_size(0);

      for (uint gid2 = 0; gid2 < ncols; gid2++) {
        uint nnz = rowNnz[gid2];
        float sum = 0;
        for (uint i = 0; i < nnz; i++) {
          uint index   = (gid2 * ellwidth) + i;
          uint col     = colIdx[index];
          float aval  = matData[index];
          float xval  = vector_x[gid*mwidth+col];
          sum  += aval * xval;
          //if (gid==0 && gid2==0)
          //  printf("aval, xval: %.2f,%.2f - %.2f: (%i,%i) \\n", aval, xval, sum, col, index);
        }
        //printf("SUM/NNZ: %.2f %i \\n", sum, nnz);
        vector_y[gid*ncols+gid2] = sum;
      }
    }""")
    ctx.save_for_backward(input, weight)

    # (isize,msize) x (msize,osize) = (isize,osize)
    matmul(ctx.cl_queue, [outshape.T[0]], None,
      weight.datat.cl, weight.idxst.cl, weight.nnzst.cl, np.uint32(weight.ellwt), np.uint32(msize), np.uint32(outshape.T[1]), input.cl, ret.data.cl)

    # resa = np.zeros(isize,osize).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, ret.cl)
    # return ret.data
    # return trans_axis(ctx, ret.data, (1,0))   # print("RES:", resa)
    return ret.data

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    topkx, topky = weight.topkx, weight.topky
    # print('BACK:', weight.shape, topkx, topky)
    isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])

    grad_input = DenseTensor(np.zeros(input.shape), dtype=np.float32)
    grad_weight = DenseTensor(np.zeros(weight.shape), dtype=np.float32)

    # print('GO:', input.shape, grad_output.shape)
    # print("OUTSHAPE:", weight.shape, input.shape[0], isize, msize, weight.ellwt)

    # grad_output = grad_output + weight.m

    matmul2 = clbuild(ctx.cl_ctx, "matmul2", """
    // DENSE x SPARSE-T
    __kernel void matmul2(__global  float* matData,     // INPUT MATRIX DATA
                            __global  uint*  colIdx,
                            __global  uint*  rowNnz,
                            uint   ellwidth,
                            uint   mwidth,
                            uint   ncols0,
                            __global  float* vector_x,    // INPUT
                            __global  float* vector_y    // OUTPUT
                            ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint nrows = get_global_size(0);
      uint nnz = rowNnz[gid];
      uint gid2 = get_global_id(1);
      uint ncols = get_global_size(1);

      float sum = 0;
      for (uint i = 0; i < nnz; i++) {
        uint index   = (gid2 * ellwidth) + i;
        uint col     = colIdx[index];
        float aval  = matData[index];
        float xval  = vector_x[gid*mwidth+col];
        sum  += aval * xval;
        //if (gid==1 && gid2==0) {
        //  printf("aval, xval: %.2f,%.2f - %.2f: (%i,%i) \\n", aval, xval, sum, col, index);
        //}
      }
      //printf("SUM/NNZ: %.2f %i \\n", sum, nnz);
      vector_y[gid*ncols+gid2] = sum;
    }""")
    # (isize,osize) x (msize,osize) = (isize,msize)
    # print('msize:', grad_output.shape, input.shape)
    matmul2(ctx.cl_queue, input.shape, None,
      weight.data.cl, weight.idxs.cl, weight.nnzs.cl, np.uint32(weight.ellw), np.uint32(grad_output.shape[1]), np.uint32(input.shape[0]), grad_output.cl, grad_input.data.cl)

    # resa = np.zeros((input.shape[1], input.shape[0])).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, grad_input.data.cl)

    # print('INPUT', DenseTensor(input).cpu().data, weight.shape[0], weight.shape[1])
    # print('OUT:', grad_input.cpu().data)




    gettopkx = clbuild(ctx.cl_ctx, "gettopkx", """
    // multilplies x TRANSPOSED by y (dense-dense)
    __kernel void gettopkx(__global  float* x,      // INPUT MATRIX DATA
                          __global  float* xsum,    // INPUT
                          __global  uint*  youtidx, // OUT
                          uint topky,
                          uint msize
                          ) { // LOCAL SHARED BUFFER
      uint isize = get_global_size(0);
      int gidx = get_global_id(0); // row

      // get topk
      xsum[gidx] = 0;
      for (uint i=0; i<msize; i++) {
        float val = x[i*isize+gidx];
        //if (gid == 0) {
        //  printf("\\nADD VALx: %.2f - %i", val, i*msize+gid);
        //}
        xsum[gidx] += val;
      }

      float valx = xsum[gidx];
      uint posx = 0;
      for (uint i = 0; i < isize; i++) {
        float tempval = fabs(xsum[i]);
        bool larger = (tempval > fabs(valx)) || (fabs(tempval) == fabs(valx) && i < gidx);
        posx += (larger)?1:0;
      }
      if (posx < topky) {
        youtidx[posx] = gidx;
      }
    }""")

    gettopky = clbuild(ctx.cl_ctx, "gettopky", """
    // multilplies x TRANSPOSED by y (dense-dense)
    __kernel void gettopky(__global  float* y,      // INPUT
                          __global  float* ysum,    // INPUT
                          __global  uint*  xoutidx, // OUT
                          uint topkx,
                          uint msize
                          ) { // LOCAL SHARED BUFFER
      uint osize = get_global_size(0);
      int gidy = get_global_id(0); // row

      ysum[gidy] = 0;
      for (uint i=0; i<msize; i++) {
        float val = y[i*osize+gidy];
        ysum[gidy] += val;
      }
      //barrier(CLK_GLOBAL_MEM_FENCE);
      float valy = ysum[gidy];
      uint posy = 0;
      for (uint i = 0; i < osize; i++) {
        float tempval = fabs(ysum[i]);
        bool larger = (tempval > fabs(valy)) || (fabs(tempval) == fabs(valy) && i < gidy);
        posy += (larger)?1:0;
      }
      if (posy < topkx) {
        xoutidx[posy] = gidy;
      }
    }""")

    sortuints = clbuild(ctx.cl_ctx, "sortuints", """
    // multilplies x TRANSPOSED by y (dense-dense)
    __kernel void sortuints(__global  uint* x,      // INPUT MATRIX DATA
                            __global  uint* xs      // INPUT
                            ) { // LOCAL SHARED BUFFER
      uint isize = get_global_size(0);
      int gidx = get_global_id(0); // row

      uint val = x[gidx];
      uint posx = 0;
      for (uint i = 0; i < isize; i++) {
        uint tempval = x[i];
        bool smaller = tempval < val;
        posx += (smaller)?1:0;
      }
      xs[posx] = x[gidx];
    }""")

    matmul0 = clbuild(ctx.cl_ctx, "matmul0", """
    // multilplies x TRANSPOSED by y (dense-dense)
    __kernel void matmul0(__global  float* x,      // INPUT MATRIX DATA
                          __global  float* y,      // INPUT
                          __global  uint* xidx,   // INPUT YIDX
                          __global  uint* yidx,   // INPUT YIDX
                          __global  float* resdata,// OUT
                          __global  uint*  rescols,
                          __global  uint*  resnnzs,
                          uint topkx,
                          uint ellw,
                          uint isize,
                          uint msize,
                          uint osize
                          ) { // LOCAL SHARED BUFFER

      uint topky = get_global_size(0);
      uint gidx = yidx[get_global_id(0)]; // row
      for (uint gidy0 = 0; gidy0 < topkx; gidy0++) {
        uint gidy = xidx[gidy0];
        float ret = 0.0;
        uint i;
        for (i = 0; i < msize; i++) {
          uint xidx = i*isize+gidx;
          float xval = x[xidx];
          uint yidx = osize*i+gidy;
          float yval = y[yidx];
          ret += xval*yval;
          //if (gidx==0 && gidy==0)
          //  printf("\\nmult: %.2f x %.2f - %.2f  -- %i/%i", xval, yval, ret, xidx, yidx);
        }
        //if (gidx==0&&gidy==0)
        //  printf("\\nsum:%.2f", ret);

        // add for
        uint nnz = resnnzs[gidx];
        for (i = 0; i < nnz; i++) {
          if (rescols[i] >= gidy) {
            break;
          }
          for (uint j = nnz; j >= i; j--) {
            //resdata[j+1] = resdata[j];
          }
        }
        resdata[gidx * ellw + gidy0] = ret;
        rescols[gidx * ellw + gidy0] = gidy;
        resnnzs[gidx] += 1;
      }
    }""")

    matmul0t = clbuild(ctx.cl_ctx, "matmul0t", """
    // multilplies x TRANSPOSED by y (dense-dense)
    __kernel void matmul0t(__global  float* x,      // INPUT MATRIX DATA
                          __global  float* y,      // INPUT
                          __global  uint* xidx,   // INPUT YIDX
                          __global  uint* yidx,   // INPUT YIDX
                          __global  float* resdata,// OUT
                          __global  uint*  rescols,
                          __global  uint*  resnnzs,
                          uint topky,
                          uint ellw,
                          uint isize,
                          uint msize,
                          uint osize
                          ) { // LOCAL SHARED BUFFER
      uint topkx = get_global_size(0);
      uint gidy = xidx[get_global_id(0)]; // row
      for (uint gidx0 = 0; gidx0 < topky; gidx0++) {
        uint gidx = yidx[gidx0];
        float ret = 0.0;
        uint i;
        for (i = 0; i < msize; i++) {
          uint xidx = i*isize+gidx;
          float xval = x[xidx];
          uint yidx = osize*i+gidy;
          float yval = y[yidx];
          ret += xval*yval;
          //if (gidx==0 && gidy==0)
          //  printf("\\nmult: %.2f x %.2f - %.2f  -- %i/%i", xval, yval, ret, gidx, gidy,i);
        }
        //if (gidx==0&&gidy==0)
        //  printf("\\nsum:%.2f", ret);

        // add for
        uint nnz = resnnzs[gidx];
        for (i = 0; i < nnz; i++) {
          if (rescols[i] >= gidy) {
            break;
          }
          for (uint j = nnz; j >= i; j--) {
            //resdata[j+1] = resdata[j];
          }
        }
        resdata[gidy * ellw + gidx0] = ret;
        rescols[gidy * ellw + gidx0] = gidx;
        resnnzs[gidy] += 1;
      }
    }""")

    # Weight update
    isize = weight.shape[0]
    msize = grad_output.shape[0]
    osize = weight.shape[1]

    dim1 = weight.shape[1]#min(weight.shape[1], topkx)
    dim2 = weight.shape[0]#min(weight.shape[0], topky)

    x_sum_buf   = DenseTensor(np.zeros(weight.shape[0]))
    y_sum_buf   = DenseTensor(np.zeros(weight.shape[1]))
    x_idx_buf   = DenseTensor(np.zeros(topkx), dtype=np.uint32)
    y_idx_buf   = DenseTensor(np.zeros(topky), dtype=np.uint32)
    xs_idx_buf  = DenseTensor(np.zeros(topkx), dtype=np.uint32)
    ys_idx_buf  = DenseTensor(np.zeros(topky), dtype=np.uint32)
    sdata_buf   = DenseTensor(np.zeros(weight.shape[0]*topkx))
    sidxs_buf   = DenseTensor(np.zeros(weight.shape[0]*topkx), dtype=np.uint32)
    snnzs_buf   = DenseTensor(np.zeros(weight.shape[0]), dtype=np.uint32)
    sdatat_buf  = DenseTensor(np.zeros(weight.shape[1]*topky))
    sidxst_buf  = DenseTensor(np.zeros(weight.shape[1]*topky), dtype=np.uint32)
    snnzst_buf  = DenseTensor(np.zeros(weight.shape[1]), dtype=np.uint32)

    # print('IN', DenseTensor(input).cpu().data, weight.shape, input.shape[0])
    # print('INPUT', grad_output.cpu().data)
    # print('OUT', grad_input.cpu().data.sum())

    # print('asdf:', isize, msize, osize)
    gettopkx(ctx.cl_queue,  [isize], None, input.cl, x_sum_buf.data.cl,
      y_idx_buf.data.cl, np.uint32(topky), np.uint32(msize))
    gettopky(ctx.cl_queue,  [osize], None, grad_output.cl,
      y_sum_buf.data.cl, x_idx_buf.data.cl, np.uint32(topkx), np.uint32(msize))
    sortuints(ctx.cl_queue,  [topkx], None, x_idx_buf.data.cl, xs_idx_buf.data.cl)
    sortuints(ctx.cl_queue,  [topky], None, y_idx_buf.data.cl, ys_idx_buf.data.cl)

    matmul0(ctx.cl_queue,  [topky], None, input.cl, grad_output.cl, xs_idx_buf.data.cl,
      ys_idx_buf.data.cl, sdata_buf.data.cl, sidxs_buf.data.cl, snnzs_buf.data.cl,
      np.uint32(topkx), np.uint32(topkx), np.uint32(isize), np.uint32(msize), np.uint32(osize))
    matmul0t(ctx.cl_queue, [topkx], None, input.cl, grad_output.cl, xs_idx_buf.data.cl,
      ys_idx_buf.data.cl, sdatat_buf.data.cl, sidxst_buf.data.cl, snnzst_buf.data.cl,
      np.uint32(topky), np.uint32(topky), np.uint32(isize), np.uint32(msize), np.uint32(osize))

    # x_sum_buf.data.cl.release()
    # y_sum_buf.data.cl.release()
    # sdata_buf.data.cl.release()
    # sidxs_buf.data.cl.release()
    # snnzs_buf.data.cl.release()
    # sdatat_buf.data.cl.release()
    # sidxst_buf.data.cl.release()
    # snnzst_buf.data.cl.release()
    # x_idx_buf.data.cl.release()
    # y_idx_buf.data.cl.release()

    newdata = {
     'data': sdata_buf.data,
     'idxs': sidxs_buf.data,
     'nnzs': snnzs_buf.data,
     'ellw': topkx,
     'datat': sdatat_buf.data,
     'idxst': sidxst_buf.data,
     'nnzst': snnzst_buf.data,
     'ellwt': topky,
    }

    w_grad = SparseTensor(from_datas=newdata, shape=weight.shape)
    # gradpy = w_grad.to_numpy()
    # print('grad_max:', w_grad.shape, gradpy.sum())
    # gradpy = w_grad.to_numpy(dual=True)
    # print('grad_max:', w_grad.shape, gradpy.sum())
    # asdf

    # updatem = clbuild(ctx.cl_ctx, "updatem", """
    # // sorts x and y in ascending order and returns sorted indices
    # __kernel void updatem(__global  float* m,     // INPUT MATRIX DATA
    #                       __global  float* grad,     // INPUT MATRIX DATA
    #                       uint msize,
    #                       uint osize,
    #                       uint topk,
    #                       float scale,
    #                       __global  uint*  xoutidx,
    #                       __global  uint*  youtidx,
    #                       __global  float* matData,     // OUTPUT MATRIX DATA
    #                       __global  uint*  colIdx,
    #                       __global  uint*  rowNnz
    #                       ) {
    #   uint gid = get_global_id(0);
    #   uint nnz = rowNnz[gid];
    #   for (uint i=0; i<nnz; i++) {
    #     uint col = colIdx[gid*topk+i];
    #     float val = matData[gid*topk+i];
    #     m[osize*gid+col] = 0;
    #   }
    #   for (uint i=0; i<osize; i++) {
    #     m[osize*gid+i] = scale * grad[osize*gid+i];
    #   }
    # }""")
    # scale = 0.9
    # updatem(ctx.cl_queue, [grad_output.shape[0],], None,
    #   weight.m.data.cl, grad_output.data.cl, np.uint32(grad_input.shape[-1]), np.uint32(grad_output.shape[1]), np.uint32(topky), np.float32(scale), xs_idx_buf.data.cl, ys_idx_buf.data.cl,
    #   sdata_buf.data.cl, sidxs_buf.data.cl, snnzs_buf.data.cl)


    return w_grad, grad_input


def trans_axis(ctx, inp, order=(1,0)):
  osize = np.array(inp.shape)[list(order)]
  ret = buffer_new(ctx, osize)
  trans = clbuild(ctx.cl_ctx, "trans", """
    __kernel void trans(__global float *a_g,
                        __global float *res_g,
                        uint width) {
      int row = get_global_id(0);
      for(uint i=0; i<width; i++) {
        //printf("\\nSET:%i-%i", row, i);
        res_g[row*width+i] = 0;
      }
    }""")
  trans(ctx.cl_queue, [osize[1]], None, inp.cl, ret.cl, np.uint32(osize[0]))

  print("PERM RET:", ret)
  return ret


class Conv2D(SparseFunction):
  def forward(ctx, x, w, stride=1, groups=1):
    if isinstance(ctx.stride, int): ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    if cin*ctx.groups != cin_: raise Exception(f"Input Tensor shape {x.shape} does not match the shape of the weights {w.shape}. ({cin*ctx.groups} vs. {cin_})")
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    ctx.save_for_backward(x,w)

    # output buffer
    ret = buffer_new(ctx, (bs, cout, oy, ox))

    # input  = (bs, groups, cin, iy, ix)
    # weight = (groups, rcout, cin, H, W)
    # output = (bs, groups, rcout, oy, ox)

    conv = clbuild(ctx.cl_ctx, "conv", """
    __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs) {

      int B = get_global_id(0)/(groups*rcout);  // range 0-bs
      int g = (get_global_id(0)/rcout)%groups;
      int c = get_global_id(0) % rcout;

      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;

      float acc = 0.0;
      for (int ci = 0; ci < cin; ci++) {
        for (int y = IY; y < IY+H; y++) {
          for (int x = IX; x < IX+W; x++) {
            acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] * \
              weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
          }
        }
      }
      output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
    }""")

    conv(ctx.cl_queue, [bs*groups*rcout, oy, ox], None,
      x.cl, w.cl, ret.cl,
      i32(H), i32(W), i32(groups), i32(rcout), i32(cin),
      i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs)
    )
    return ret

  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    dx = buffer_new(ctx, (bs, cin_, iy, ix), zero=True)
    dw = buffer_new(ctx, (cout, cin, H, W))

    # tensx = (bs, groups*cin, iy, ix)
    # tensw = (groups*rcout, cin, H, W)
    # ggg = (bs, groups*rout, oy, ox)

    convw = clbuild(ctx.cl_ctx, "convw", """
    __kernel void convw(__global const float *tensx, __global const float *ggg, __global float *dw,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int g = get_global_id(0)/(rcout*cin) ; // range 0-groups
      int c = (get_global_id(0)/(cin)) %rcout; // range 0-rcout
      int ci = get_global_id(0) % cin;        // range 0-cin
      int y = get_global_id(1);  // range 0-H
      int x = get_global_id(2);  // range 0-W

      float acc = 0.0;
      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int B = 0; B < bs; B++) {
            acc += ggg[B*groups*rcout*oy*ox + +g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
              tensx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x];
          }
        }
      }
      dw[get_global_id(0)*H*W + y*W + x] = acc;
    }""")
    convx = clbuild(ctx.cl_ctx, "convx", """
    __kernel void convx(__global const float *tensw, __global const float *ggg, __global float *dx,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int B = get_global_id(0);
      int g = get_global_id(1);
      int ci = get_global_id(2);

      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
              float acc = 0.0;
              for (int c = 0; c < rcout; c++) {
                acc += ggg[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
                  tensw[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
              }
              dx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x] += acc;
            }
          }
        }
      }
    }
    """)

    conv_args = i32(H), i32(W), i32(ctx.groups), i32(rcout), i32(cin), i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs), i32(bs)
    convw(ctx.cl_queue, [ctx.groups*rcout*cin, H, W], None, x.cl, grad_output.cl, dw.cl, *conv_args)
    convx(ctx.cl_queue, [bs, ctx.groups, cin], None, w.cl, grad_output.cl, dx.cl, *conv_args)
    return dx, dw
