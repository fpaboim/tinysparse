import functools
import pyopencl as cl
import numpy as np
from .sparsetensor import SparseFunction, topk
from .densetensor import GPUBuffer

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
    ctx.save_for_backward(x.shape)
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    r = GPUBuffer(shape, hostbuf=x)   # NOTE: this is not a copy
    assert np.prod(x.shape) == np.prod(r.shape)
    return r

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return GPUBuffer(in_shape, hostbuf=grad_output)

def perm_axis(ctx, inp, order):
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
  return ret

class Transpose(SparseFunction):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    return perm_axis(ctx, x, order)

  def backward(ctx, grad_output):
    return perm_axis(ctx, grad_output, np.argsort(ctx.order))

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
  def forward(ctx, input, weight):
    # print("WEGHT:", input.shape, weight.shape)
    BS = weight.shape[0]
    # print('sprmult:', input, weight)
    # assert input.shape[-1] == weight.shape[-2]
    # cnt = 1#np.prod(input.shape[0:-2]) if len(input.shape) > 2 else 1
    # isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])
    outshape = (BS, input.shape[0])
    ret = buffer_new(ctx.cl_ctx, outshape, zero=True)
    # print("RET:", ret)
    # print("RET:", weight)

    matmul = clbuild(ctx.cl_ctx, "matmul", """
    // Every global_id_0 works on a row
    __kernel void matmul(__global  float* matData,     // INPUT MATRIX DATA
                            __global  uint*  colIdx,
                            __global  uint*  rowNnz,
                            uint   ellwidth,
                            __global  float* vector_x,    // INPUT
                            __global  float* vector_y    // OUTPUT
                            ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint nrows = get_global_size(0);
      uint gid2 = get_global_id(1);

      uint nnz    = rowNnz[gid];
      uint baseidx = gid2*nrows;
      float sum = 0;
      for (uint i = 0; i < nnz; i++) {
        uint index   = (gid * ellwidth) + i;
        uint col     = colIdx[index];
        float aval  = matData[index];
        float xval  = vector_x[baseidx+col];
        //printf("aval, xval: %.2f,%.2f: (%i,%i) \\n", aval, xval, col, index);
        sum  += aval * xval;
      }
      //printf("SUM/NNZ: %.2f %i \\n", sum, nnz);
      vector_y[baseidx+gid] = sum;
    }""")
    ctx.save_for_backward(input, weight, matmul)

    # (isize,msize) x (msize,osize) = (isize,osize)
    matmul(ctx.cl_queue, [input.shape[0], weight.shape[0]], None,
      input.data.cl, input.idxs.cl, input.nnzs.cl, np.uint32(input.ellw), weight.data.cl, ret.cl)

    # resa = np.zeros(weight.shape).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, ret.cl)
    # print("RES:", resa)

    # print("LEN:", weight)
    # print('IN:', resa)
    # print("RET:", ret)
    return ret

  def backward(ctx, grad_output):
    input, weight, matmul = ctx.saved_tensors
    # print("BACK:", input.shape, weight.shape)

    # print('asdf:', ctx, ctx.cl_queue, ctx.cl_ctx, (weight.shape[0]))
    grad_input = buffer_new(ctx, weight.shape)
    grad_weight = buffer_new(ctx, (topk*topk*weight.shape[0],))
    # grad_weight = cl.Buffer(ctx.cl_ctx, cl.mem_flags.READ_WRITE, weight.shape[0]*topk*topk*4)

    # Grad update
    # (isize,osize) x (msize,osize) = (isize,msize)
    matmul(ctx.cl_queue, [input.shape[0], weight.shape[0]], None,
      input.datat.cl, input.idxst.cl, input.nnzst.cl, np.uint32(input.ellwt), grad_output.cl, grad_input.cl)

    # resa = np.ones(weight.shape).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, weight.data.cl)
    # print('x:', resa.shape)
    # resa = np.ones(weight.shape).astype(np.float32)
    # cl.enqueue_copy(ctx.cl_queue, resa, weight.data.cl)
    # print('y:', resa)

    # Weight update
    x_idx_buf = buffer_new(ctx, (weight.shape[0], topk), dtype=np.uint32, zero=True)
    y_idx_buf = buffer_new(ctx, (weight.shape[0], topk), dtype=np.uint32, zero=True)

    genwupdate2 = clbuild(ctx.cl_ctx, "genwupdate2", """
    // sorts x and y in ascending order and returns sorted indices
    __kernel void genwupdate2(__global  float* x,     // INPUT MATRIX DATA
                             __global  float* y,    // INPUT
                             __global  float* xout,    // INPUT
                             uint topk,
                             uint bs,
                             __global  uint* xoutidx,    // INPUT
                             __global  uint* youtidx    // INPUT
                            ) { // LOCAL SHARED BUFFER
      uint gid = get_global_id(0);
      uint n = get_global_size(0);
      //uint bs = get_global_size(1);
      //uint gid2 = get_global_id(1);

      for (uint gid2=0; gid2<bs; gid2++){
        uint idx = n*gid2+gid;

        float valx = x[idx];
        float valy = y[idx];
        uint posx = 0;
        uint posy = 0;
        for (uint i = 0; i < n; i++) {
          uint idx2 = n*gid2+i;
          float tempval = x[idx2];
          float tempval2 = y[idx2];
          bool larger = tempval > valx;
          bool larger2 = tempval2 > valy;

          posx += (larger)?1:0;
          posy += (larger2)?1:0;
        }
        //printf("posx:%i", posx);
        if (posx < topk) {
          xoutidx[posx+topk*gid2] = gid;
          //printf("\\nxoutidx[%i]:%i", posx, gid);
        }
        if (posy < topk) {
          youtidx[posy+topk*gid2] = gid;
        }
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
      for (uint gid2=0; gid2<bs; gid2++){
        if (gid < topk) {
          for (uint j=0; j<topk; j++) {
            float res = x[xoutidx[gid+topk*gid2]+gid2*n] * y[youtidx[j+topk*gid2]+gid2*n];
            //printf("\\nJ:%i  gid:%i idx:%i / %.2f", j, gid, xoutidx[gid+topk*gid2], y[youtidx[j+topk*gid2]+gid2*n]);
            //printf("\\nRES:%.2f - %i - %i -  %.2f - %.2f",res, xoutidx[gid+topk*gid2], youtidx[j+topk*gid2], x[xoutidx[gid+topk*gid2]+gid2*n], y[youtidx[j+topk*gid2]+gid2*n]);
            //barrier(CLK_GLOBAL_MEM_FENCE);
            xout[gid2*topk*topk+j*topk+gid] = res;
          }
        }
      }
    }""")

    # weight: bs x rows

    # print('GRTRTS:', weight.shape, grad_output.shape, topk)
    genwupdate2(ctx.cl_queue, [weight.shape[1]], None,
      weight.data.cl, grad_output.cl, grad_weight.cl, np.uint32(topk), np.uint32(weight.shape[0]), x_idx_buf.cl, y_idx_buf.cl)

    # gen alt data structure to add grad evetually in the optimizer
    # print("RAN")

    # bs = weight.shape[0]
    # resx = np.zeros(bs*topk*topk).astype(np.float32)
    # resxidx = np.zeros(bs*topk).astype(np.uint32)
    # resyidx = np.zeros(bs*topk).astype(np.uint32)

    # cl.enqueue_copy(ctx.cl_queue, resx, grad_weight.cl)
    # cl.enqueue_copy(ctx.cl_queue, resxidx, x_idx_buf.cl)
    # cl.enqueue_copy(ctx.cl_queue, resyidx, y_idx_buf.cl)
    # print('GW0:', resx)

    return (grad_weight, x_idx_buf, y_idx_buf), grad_input

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
