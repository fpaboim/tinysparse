import numpy as np

class Device: CPU, GPU, ANE = 0, 1, 2

DEFAULT_DEVICE = Device.GPU

class Tensor:
  def __init__(self, data, device=DEFAULT_DEVICE, requires_grad=True):
    pass

  def is_sparse(self):
    return False

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx:
        [_deepwalk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  # ***** toposort and backward pass *****
