def iterate(x):
    data = []
    for ele in x:
        if isinstance(ele, list):
            data.extend(iterate(ele))
        else:
            data.append(ele)

    return data


class Tensor:
    def __init__(self, data=[]):
        self.data = iterate(data)
        self.og = data
        dim = []
        x = data
        while isinstance(x, list) and len(x) > 0:
            dim.append(len(x))
            x = x[0]

        self.dim = tuple(dim)
        stride = [1]
        s = 1
        for e in reversed(dim[1:]):
            s *= e
            stride.append(s)
        self.stride = tuple(reversed(stride))

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
x = Tensor(data)
print(x)
print(x.dim)
print(x.stride)
print(x.ind((0, 2)))
print(x.slice(2, 0))
