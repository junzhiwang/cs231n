from layers import *
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from time import time

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

np.random.seed(231)
x = np.random.randn(100, 3, 500, 500)
dout = np.random.randn(100, 3, 250, 250)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_forward_fast(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_general(x, pool_param)
t2 = time()

print('Testing pool_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))

t0 = time()
dx_naive = max_pool_backward_fast(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_general(dout, cache_fast)
t2 = time()

print('\nTesting pool_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
