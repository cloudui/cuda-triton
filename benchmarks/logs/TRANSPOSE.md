# Perf profiles

## Naive Transpose
```cuda
out[y][x] = in[x][y]
```

```
root@45c8f1499741:/# python benchmarks/bench_cuda_transpose.py 
MxN              PyTorch (ms)    CUDA (ms)      CUDA vs PyT
------------------------------------------------------------
128x128          0.0043          0.0042         1.02x
512x512          0.0129          0.0092         1.41x
1024x1024        0.0293          0.0227         1.29x
2048x2048        0.0820          0.0592         1.39x
4096x4096        0.3344          0.1956         1.71x
1024x4096        0.0884          0.0575         1.54x
```

## Coalesced with SMEM staging
```cuda
smem[y][x] = in[x][y]

out[x'][y'] = smem[x][y]
```

```
root@45c8f1499741:/# python benchmarks/bench_cuda_transpose.py 
MxN              PyTorch (ms)    CUDA (ms)      CUDA vs PyT
------------------------------------------------------------
128x128          0.0043          0.0047         0.91x
512x512          0.0126          0.0072         1.76x
1024x1024        0.0294          0.0191         1.54x
2048x2048        0.0820          0.0568         1.44x
4096x4096        0.3336          0.1924         1.73x
1024x4096        0.0881          0.0560         1.57x
```

## Coalesced and Padded for bank conflict
```
python benchmarks/bench_cuda_transpose.py 
MxN              PyTorch (ms)    CUDA (ms)      CUDA vs PyT
------------------------------------------------------------
128x128          0.0043          0.0043         1.00x
512x512          0.0127          0.0067         1.88x
1024x1024        0.0294          0.0192         1.53x
2048x2048        0.0820          0.0565         1.45x
4096x4096        0.3329          0.1922         1.73x
1024x4096        0.0882          0.0558         1.58x
16384x16384      5.2502          3.1881         1.65x
```