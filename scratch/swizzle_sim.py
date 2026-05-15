"""
Pure-Python simulator for CuTe's Swizzle<B, M, S> bit-permutation. Lets you
toy with swizzle parameters and verify bank assignments without compiling a
CUDA kernel. Used in the blog's "Swizzling" section for readers who want to
play with the bit math.

Run:
  python scratch/swizzle_sim.py

Tweak M, N, SIZE, and the Swizzle<B, M, S> at the bottom to explore
different access patterns. `Type.get_bank(i, j)` returns the SMEM bank that
element (i, j) maps to.
"""


def get_bank(addr):
    """SMEM bank for a byte address, assuming 4-byte banks across 32 banks."""
    return (addr // 4) % 32


class Swizzle:
    def __init__(self, B, M, S):
        self.B = B
        self.M = M
        self.S = S

    def __call__(self, offset):
        return self.swizzle(offset)

    def swizzle(self, offset):
        B, M, S = self.B, self.M, self.S
        mid_bits = offset >> M
        col = mid_bits & ((1 << B) - 1)
        rest_mid = mid_bits & ~((1 << B) - 1)
        row = offset >> (M + S)
        newB = row ^ col

        return (row << (M + S)) + ((rest_mid | newB) << M) + (offset & ((1 << M) - 1))


class Type:
    def __init__(self, M, N, size, swizzle=None):
        self.M = M
        self.N = N
        self.size = size
        self.stride_row = size * N
        self.stride_col = size
        self.swizzle = swizzle
        self.vals = {}

    def get_addr(self, i, j):
        offset = i * self.N + j
        if self.swizzle:
            offset = self.swizzle(offset)
        return offset * self.size

    def get_bank(self, i, j):
        return (self.get_addr(i, j) // 4) % 32

    def set_swizzle(self, B, M, S):
        self.swizzle = Swizzle(B, M, S)

    def put(self, i, j, val):
        addr = self.get_addr(i, j)
        self.vals[addr] = val

    def get(self, i, j):
        addr = self.get_addr(i, j)
        return self.vals[addr]


if __name__ == "__main__":
    # Mock a (16, 32) fp16 SMEM tile (SIZE=2 bytes/element) with FA2's
    # Swizzle<2, 3, 2>. Walk the first 8 rows x first 4 columns of 16-bit
    # pairs and print which bank each lands in. With a correct swizzle, no
    # row should hit the same bank twice.
    SIZE = 2
    M, N = 16, 32

    t = Type(M, N, SIZE)
    t.set_swizzle(2, 3, 2)

    for i in range(M):
        for j in range(N):
            t.put(i, j, (i, j))

    banks = []
    correct = 0
    count = 0
    for i in range(8):
        for j in range(0, 8, 2):
            bank = t.get_bank(i, j)
            banks.append(bank)
            val = t.get(i, j)
            print(f"({i}, {j}): {val}, bank: {bank}")
            if val == (i, j):
                correct += 1
            count += 1

    print()
    print(f"bank accesses: {len(banks)}, distinct banks: {len(set(banks))}")
    print(f"{correct} / {count} retrieved correctly")
