import numpy as np

def _init_orthogonal(shape):
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim

    num_cols = shape[-1]
    if num_rows < num_cols:
      flat_shape = (num_cols, num_rows)
    else:
      flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = np.random.normal(size = flat_shape)
    # Compute the qr factorization
    q, r = np.linalg.qr(a)
    # Make Q uniform
    q *= np.sign(np.diag(r))
    if num_rows < num_cols:
      q = q.T

    return q.reshape(shape)

_INITIALIZERS = {
    'zeros': lambda x: np.zeros(x),
    'ones': lambda x: np.ones(x),
    'identity': lambda x: np.pad(np.eye(min(x)), [(0, x[i] - min(x)) for i in range(len(x))], 'constant'),
    'uniform': lambda x: np.random.uniform(size = x),
    # glorot_uniform assumes weight shape to be (#input tensors,...,#output tensors)
    'glorot_uniform': lambda x: np.random.uniform(-np.sqrt(6 / sum([x[0], x[-1]])), np.sqrt(6 / sum([x[0], x[-1]])), size = x),
    'orthogonal': _init_orthogonal,
    'normal': lambda x: np.random.normal(size = x)
}

def get(identifier):
    if identifier in _INITIALIZERS:
        return _INITIALIZERS[identifier]
    else:
        raise ValueError('Could not interpret initializer identifier: ' + str(identifier))

