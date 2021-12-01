### TinySparse
This is a fork of tinygrad (https://github.com/geohot/tinygrad) made to work with sparse matrices end to end, start to finish. I'll eventually, hopefully, write something explaining the process. Matrices are stored in ellkpack format and sparse backpropagation is performed by my topk sparse backprop algorithm.

### Installation

```bash
pip3 install git+https://github.com/fpaboim/tinysparse.git --upgrade
```

### Example

```python
from tinygrad.densetensor import DenseTensor

x = DenseTensor.eye(3)
y = DenseTensor([[2.0,0,-2.0]])
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

## Sparse Tensors
```bash
python3 examples/serious_mnist.py
```

