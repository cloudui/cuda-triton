"""
WIP toy 2D `Tensor` with row-major and column-major layouts in pure Python.
Useful for sanity-checking row/col-major + stride math from the blog's
"Layout Hell" section. Not generalized -- 2D only, integer values only,
written for ad-hoc poking.

Run:
  python scratch/tensor.py
"""

import numpy as np


def iterate(x):
    data = []
    for ele in x:
        if isinstance(ele, list):
            data.extend(iterate(ele))
        else:
            data.append(ele)

    return data


class Tensor:
    def __init__(self, data=[], col_major=False):
        # self.data = iterate(data)
        a = np.array(data)
        data = []
        if col_major:
            for j in range(a.shape[1]):
                data += list(map(int, a[:, j]))
            self.data = data
        else:
            for i in range(a.shape[0]):
                data += list(map(int, a[i, :]))
            self.data = data
        self.og = data
        self.dim = a.shape
        dim = self.dim
        # x = data
        # # while isinstance(x, list) and len(x) > 0:
        # #     dim.append(len(x))
        # #     x = x[0]

        # self.dim = tuple(dim)
        stride = [1]
        s = 1
        it = dim[:-1] if col_major else reversed(dim[1:])
        for e in it:
            s *= e
            stride.append(s)
        self.stride = tuple(stride) if col_major else tuple(reversed(stride))

    def __str__(self):
        return f"tensor({str(self.og)})"

    def ind(self, ind):
        idx = 0
        for stride, dimind in zip(self.stride, ind):
            idx += dimind * stride

        return self.data[idx]

    def slice(self, dim, val):
        result = []

        def iter(dimi, idx: list):
            if dimi >= len(self.dim):
                result.append(self.ind(idx))
                return
            if dimi == dim:
                idx[dimi] = val
                iter(dimi + 1, idx)
                return
            for i in range(self.dim[dimi]):
                idx[dimi] = i
                iter(dimi + 1, idx)

        iter(0, [0] * len(self.dim))
        return result


data = [
    [[i * 8 + 2 * j + k for k in range(1, 3)] for j in range(0, 4)] for i in range(0, 4)
]
# x = Tensor(data)
# print(x)
# print(x.dim)
# print(x.stride)
# print(x.ind((0, 2)))
# print(x.slice(2, 0))

nd = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
x = Tensor(nd)
y = Tensor(nd, col_major=True)

print(x.data)
print(y.data)
print(x.stride)
print(y.stride)
print(x.ind((1, 3)))
print(y.ind((1, 3)))

for i in range(x.dim[0]):
    for j in range(x.dim[1]):
        print(x.ind((i, j)))
        print(y.ind((i, j)))
