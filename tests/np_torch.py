"""Minimal numpy stand-in for the few torch ops seg/tracker.py uses, so the mock test runs without torch.
Only for tests; the real pipeline uses torch + SAM 2."""
import contextlib
import types

import numpy as np


class TT(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def float(self):
        return self.astype(np.float32)

    def to(self, x):
        return self.astype(x) if isinstance(x, (type, np.dtype)) else self

    @property
    def device(self):
        return "cpu"

    def max(self, dim=None, *a, **k):
        arr = np.asarray(self)
        if dim is None:
            return arr.max(*a, **k)
        return arr.max(dim).view(TT), arr.argmax(dim).view(TT)


def _t(a):
    return np.asarray(a).view(TT)


torch = types.SimpleNamespace(
    full=lambda shape, v: _t(np.full(shape, v, np.float32)),
    from_numpy=_t,
    tensor=lambda x, device=None: _t(x),
    where=lambda c, a, b: _t(np.where(np.asarray(c), np.asarray(a), np.asarray(b))),
    zeros_like=lambda a: _t(np.zeros_like(np.asarray(a))),
    sigmoid=lambda x: _t(1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))),
    bincount=lambda x, weights=None, minlength=0: _t(np.bincount(
        np.asarray(x).astype(np.int64).ravel(), None if weights is None else np.asarray(weights).ravel(), minlength)),
    int32=np.int32, bfloat16=None,
    autocast=lambda *a, **k: contextlib.nullcontext(),
    no_grad=contextlib.nullcontext, inference_mode=contextlib.nullcontext,
)
