# # for i in range(0, 15, 4):
# #     # print(f'og bank: {i // 4 % 32}')
# #     row = i // 128
# #     col = i // 4
# #     print(f"address: {i}")
# #     print(f"fixed address: {row ^ col}")


# # print()
# # for i in range(0, 15, 4):
# #     i += 128
# #     row = i // 128
# #     col = i // 4
# #     print(f"address: {i}")
# #     print(f"fixed address: {row ^ col}")
def get_bank(addr):
    return (addr // 4) % 32


# def swizzle(row_shift, col_shift):
#     banks = set()
#     mem = []
#     bank_conflicts = 0
#     mem_conflicts = 0
#     for i in range(8):
#         i += row_shift
#         row_start = i * 128
#         bs = row_start + col_shift * 16
#         s, e = bs, bs + 15
#         print(f"Original address: {s}-{e}")
#         print(f"Banks: {get_bank(s)}-{get_bank(e)}")
#         print("-" * 10)
#         row = bs >> 7
#         col = (bs >> 3) & 0b111
#         perm = row ^ col

#         ns = row_start + perm * 16
#         ne = ns + 15

#         for i in range(get_bank(ns), get_bank(ne) + 1):
#             if i in banks:
#                 bank_conflicts += 1
#             banks.add(i)
#         for i in range(ns, ne + 1):
#             mem.append(i)

#         print(ne, ns)
#         print(f"row: {row}, col: {col}, shift: {perm}")
#         print(f"New address: {ns}-{ne}")
#         print(f"New banks: {get_bank(ns)}-{get_bank(ne)}")
#         print(">" * 10)
#         print()
#     return bank_conflicts, mem


# # if __name__ == "__main__":
# #     mem = set()
# #     ac = 0
# #     bank_conflicts = 0
# #     for row in range(1):
# #         for col in range(1, 2):
# #             bc, m = swizzle(row, col)
# #             bank_conflicts += bc

# #             for addr in m:
# #                 if addr in mem:
# #                     ac += 1
# #                 mem.add(addr)
# #             print(bc)

# #     print("-" * 20)
# #     print(bank_conflicts, ac)

# # for i in range(0, 128, 16):
# #     bank = get_bank(i)
# #     bs = i
# #     # print(f"banks: {bank}-{bank+3}")
# #     row = bs >> 7
# #     col = (bs >> 3) & 0b111
# #     perm = row ^ col

# #     na = perm * 16 + bs
# #     bank = get_bank(na)
# #     print(f"banks: {bank}-{bank+3}")
# #     # print()
# i = 7
# j = 38
# col = ((i * 128 + j) >> 3) & 0b111
# perm = i ^ col
# print(i * 128 + 16 * perm + 38 % 16)


class Swizzle:
    def __init__(self, B, M, S):
        self.B = B
        self.M = M
        self.S = S

    def __call__(self, offset):
        return self.swizzle(offset)

    def swizzle(self, offset):
        B, M, S = self.B, self.M, self.S
        col = (offset >> M) & ((1 << B) - 1)
        row = offset >> (M + S)
        newB = row ^ col

        return (row << (M + S)) + (newB << M) + (offset & ((1 << M) - 1))


class Type:
    def __init__(self, M, N, size, swizzle=None):
        self.M = M
        self.N = N
        self.size = size
        self.stride_row = size * N
        self.stride_col = size
        self.swizzle = swizzle

    def get_addr(self, i, j):
        offset = i * self.N + j
        if self.swizzle:
            offset = self.swizzle(offset)
        return offset * self.size

    def get_bank(self, i, j):
        return (self.get_addr(i, j) // 4) % 32

    def set_swizzle(self, B, M, S):
        self.swizzle = Swizzle(B, M, S)


SIZE = 2
M = 16
N = 64

t = Type(M, N, SIZE)
t.set_swizzle(3, 3, 3)

banks = set()
col_offset = 0
for i in range(4):
    for j in range(0, 64, 64 // 8):
        bank = t.get_bank(i, j)
        banks.add(bank)

print(banks)
conflicts = 32 // len(banks)

print(f"# bank conflicts: {conflicts}")

print(t.get_bank(1, 0))

# s = Swizzle(3, 3, 3)
# print(s.swizzle(0))
# print(s.swizzle(64))
