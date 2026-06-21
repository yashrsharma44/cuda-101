# CUDA Programming Course — Study Notes
**Source:** [CUDA Programming Course – High-Performance Computing with GPUs](https://www.youtube.com/watch?v=86FAWCzIe_4)
**Instructor:** Elliot Arledge (freeCodeCamp) | **Duration:** ~12 hours | **Code:** https://github.com/Infatoshi/cuda-course

---

## Table of Contents
1. [Deep Learning Ecosystem](#1-deep-learning-ecosystem)
2. [Setup / Installation](#2-setup--installation)
3. [C/C++ Review](#3-cc-review)
4. [Intro to GPUs](#4-intro-to-gpus)
5. [Writing Your First Kernels](#5-writing-your-first-kernels)
6. [CUDA APIs](#6-cuda-apis)
7. [Optimizing Matrix Multiplication](#7-optimizing-matrix-multiplication)
8. [Triton](#8-triton)
9. [PyTorch Extensions (CUDA)](#9-pytorch-extensions-cuda)
10. [Final Project — MNIST MLP](#10-final-project--mnist-mlp)
11. [Next Steps](#11-next-steps)

---

## 1. Deep Learning Ecosystem

### What is CUDA?
CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and API model. It exposes the GPU as a massively parallel processor for general-purpose computing (GPGPU), going far beyond graphics.

### Where CUDA fits in the stack
```
High-Level Frameworks
  PyTorch / TensorFlow / JAX
        |
   CUDA / cuDNN / cuBLAS          <-- you are here after this course
        |
   NVIDIA Driver (host-injected)
        |
   GPU Hardware (CUDA cores, Tensor Cores, etc.)
```

### Why GPUs for Deep Learning?
| CPU | GPU |
|-----|-----|
| Few (4–128) powerful cores | Thousands of simpler cores |
| Optimized for serial/branch-heavy tasks | Optimized for massively parallel math |
| Large cache, complex branch predictor | High memory bandwidth, SIMT |
| Best for: control flow, OS, I/O | Best for: matrix multiply, convolutions |

### Use cases beyond DL
- Fluid / physics simulation
- Video encoding/decoding
- Scientific computing (molecular dynamics, climate models)
- Graphics rendering

---

## 2. Setup / Installation

### Requirements
- Linux (Ubuntu preferred — course targets Ubuntu)
- NVIDIA GPU: any GTX/RTX or datacenter GPU
- Cloud alternative: vast.ai, Lambda Labs, RunPod

### Key installs
```bash
# CUDA Toolkit (match to driver_max_cuda, NOT the driver itself)
sudo apt install cuda-toolkit-12-x

# Python env
source /venv/main/bin/activate

# Verify GPU is visible
nvidia-smi
nvcc --version
```

### Compilation flow
```
foo.cu  -->  nvcc  -->  PTX (virtual ISA)  -->  SASS (native ISA)  -->  GPU executes
               |                                      ^
               +---- host (CPU) code --->  g++/clang  |
                                                       |
                                          (JIT at load time if only PTX shipped)
```

> **PTX-JIT caveat:** if a wheel ships PTX built with a *newer* toolkit than the host driver understands, you get `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Fix: use a toolkit version ≤ `driver_max_cuda`.

---

## 3. C/C++ Review

### Key concepts used throughout the course
```c
// Pointers and memory management — critical for GPU programming
int *ptr = (int*)malloc(N * sizeof(int));   // host allocation
free(ptr);

// cudaMalloc mirrors this pattern on device
float *d_arr;
cudaMalloc(&d_arr, N * sizeof(float));
cudaFree(d_arr);
```

### Structs and pass-by-reference
```c
typedef struct { float x, y, z; } Vec3;
void scale(Vec3 *v, float s) { v->x *= s; v->y *= s; v->z *= s; }
```

### Relevant C++ features
- **Templates** — used by cuBLAS/cuDNN wrappers and PyTorch extension bindings
- **`extern "C"`** — needed to expose CUDA kernels to Python/ctypes without name mangling
- **References vs pointers** — device code only supports pointers (no references to device memory from host)

---

## 4. Intro to GPUs

### GPU Architecture Overview

```
╔══════════════════════════════════════════════════════════════╗
║                         GPU DIE                             ║
║                                                              ║
║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ║
║  │   SM 0   │  │   SM 1   │  │   SM 2   │  │  SM N    │   ║
║  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │   ║
║  │ │Cores │ │  │ │Cores │ │  │ │Cores │ │  │ │Cores │ │   ║
║  │ │32–128│ │  │ │32–128│ │  │ │32–128│ │  │ │32–128│ │   ║
║  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │   ║
║  │ SharedMem│  │ SharedMem│  │ SharedMem│  │ SharedMem│   ║
║  │ Regs     │  │ Regs     │  │ Regs     │  │ Regs     │   ║
║  └──────────┘  └──────────┘  └──────────┘  └──────────┘   ║
║                                                              ║
║              L2 Cache (shared across all SMs)               ║
║                                                              ║
║         Global Memory / VRAM  (HBM2/GDDR6 — GBs)           ║
╚══════════════════════════════════════════════════════════════╝
```

### Streaming Multiprocessor (SM) internals
```
SM
├── 32–128 CUDA Cores (FP32/INT32 ALUs)
├── Tensor Cores (matrix math, FP16/BF16/INT8)
├── Warp Schedulers (2–4 per SM)
├── Register File (64K 32-bit registers)
├── Shared Memory / L1 Cache (16–100 KB, configurable split)
└── Load/Store Units, SFUs (special function units: sin, sqrt, etc.)
```

### Thread Hierarchy

```
GRID  (the entire kernel launch)
└── BLOCK 0  ── BLOCK 1  ── BLOCK 2  ── ... ── BLOCK M
      └── Thread 0                                  Each block maps
          Thread 1                                  to one SM.
          Thread 2                                  Blocks never split
          ...                                       across SMs.
          Thread 1023   (max 1024 threads/block)
```

### Warp execution (SIMT)
```
Block of 512 threads
└── Warp 0  (threads  0–31 )  ┐
    Warp 1  (threads 32–63 )  │  Each warp executes the
    Warp 2  (threads 64–95 )  │  SAME instruction in lockstep
    ...                        │  on 32 CUDA cores.
    Warp 15 (threads 480–511) ┘

If threads diverge (if/else) → warp serializes branches → AVOID THIS
```

### Memory Hierarchy (speed vs capacity)

```
Fastest ──────────────────────────────────────────────── Slowest
│ Registers │ Shared Mem │ L1 Cache │ L2 Cache │ Global Mem │ Host RAM │
│ ~1 cycle  │  ~5 cycles │ ~30 cyc  │ ~80 cyc  │  ~300 cyc  │ PCIe μs  │
│ per-thread│  per-block │ per-SM   │ per-GPU  │  all SMs   │ CPU side │
│ ~64K regs │  48–100 KB │ (same HW)│  ~10 MB  │  GBs VRAM  │  GBs RAM │
```

---

## 5. Writing Your First Kernels

### Anatomy of a CUDA kernel

```c
// __global__ = runs on GPU, callable from CPU
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // Compute unique thread ID across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check — last block may be partially filled
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host launch:
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

### Thread ID formula (2D example)

```
Grid (Bx × By blocks)
                  blockIdx.x=0  blockIdx.x=1  blockIdx.x=2
blockIdx.y=0    [  Block 00  ] [  Block 10  ] [  Block 20  ]
blockIdx.y=1    [  Block 01  ] [  Block 11  ] [  Block 21  ]

Inside Block 11 (blockDim = 16×16):
  row = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
  flat_idx = row * total_cols + col
```

### Host ↔ Device data flow

```
Host (CPU)                            Device (GPU)
──────────                            ─────────────
float *h_a (malloc)                   float *d_a (cudaMalloc)
     │                                      │
     └──── cudaMemcpy H→D ──────────────────►
                                            │
                                     kernel<<<...>>>()
                                            │
     ◄──── cudaMemcpy D→H ──────────────────┘
     │
float *h_c  (result on CPU)
```

### Synchronization
```c
kernel<<<grid, block>>>(args);  // async — returns immediately
cudaDeviceSynchronize();        // blocks host until GPU is done
// OR: cudaMemcpy implicitly synchronizes
```

### Common first kernels
1. **Vector add** — 1D grid, one element per thread
2. **Matrix transpose** — 2D grid, shared memory to avoid uncoalesced writes
3. **Reduction (sum)** — tree-style parallel sum within a block using shared mem

---

## 6. CUDA APIs

### cuBLAS — GPU BLAS (linear algebra)
```c
cublasHandle_t handle;
cublasCreate(&handle);

// SGEMM: C = alpha * A * B + beta * C
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha, d_A, M,
             d_B, K,
    &beta,  d_C, M);

cublasDestroy(handle);
```

### cuDNN — GPU Deep Neural Network primitives
- Convolutions, pooling, batch norm, activation functions
- Takes `cudnnTensorDescriptor_t` to describe tensor shapes/layouts
- PyTorch uses cuDNN under the hood for `conv2d`, `BatchNorm`, etc.

### cuFFT, cuRAND, cuSPARSE
| Library | Purpose |
|---------|---------|
| cuFFT | Fast Fourier Transforms on GPU |
| cuRAND | Random number generation |
| cuSPARSE | Sparse matrix operations |
| NCCL | Multi-GPU collective communications |

### CUDA Streams — overlapping compute and data transfer
```
Stream 0 (default):  [transfer A] ──► [kernel A] ──► [transfer result A]
Stream 1:                    [transfer B] ──► [kernel B] ──► [transfer result B]
                                      ↑ overlaps with Stream 0's kernel
```
```c
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaMemcpyAsync(d_a, h_a, sz, cudaMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s1>>>(d_a, d_c);
```

---

## 7. Optimizing Matrix Multiplication

### Naive GEMM — the starting point

```
For C[i][j] = sum_k A[i][k] * B[k][j]:

Each thread computes ONE output element.
Problem: each thread reads an entire row of A and column of B from GLOBAL memory.
→ O(N³) global memory reads = bandwidth bottleneck
```

### Optimization 1 — Tiled Shared Memory GEMM

```
Tile A (TILE×TILE)       Tile B (TILE×TILE)
loaded into              loaded into
shared memory            shared memory
by cooperating           by cooperating
block threads            block threads
      │                        │
      └──────── dot product ───┘
                    │
              accumulate into
              register (per-thread)
              then write C[i][j]

Global memory reads: N³/TILE  (e.g. 32× fewer with TILE=32)
```

```c
__global__ void tiled_matmul(float *A, float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N/TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row * N + t*TILE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        __syncthreads();  // wait for all threads to load the tile

        for (int k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();  // wait before loading next tile
    }
    if (row < N && col < N) C[row*N + col] = acc;
}
```

### Optimization 2 — Memory Coalescing

```
GOOD (coalesced): consecutive threads access consecutive memory addresses
  Thread 0 → A[0], Thread 1 → A[1], Thread 2 → A[2]  ← single 128-byte transaction

BAD (strided): threads skip around memory
  Thread 0 → A[0], Thread 1 → A[32], Thread 2 → A[64] ← 32 separate transactions
```

### Optimization 3 — Register Tiling (thread-level accumulation)
Each thread computes a small sub-tile (e.g. 4×4) of C rather than 1 element, amortizing the load overhead and increasing arithmetic intensity.

### Performance ladder
```
Naive GEMM           ~  5% of peak TFLOPS
Shared-mem tiling    ~ 30% of peak TFLOPS
+ coalescing fixes   ~ 50% of peak TFLOPS
+ register tiling    ~ 70–80% of peak TFLOPS
cuBLAS               ~ 90–95% of peak TFLOPS
```

---

## 8. Triton

### What is Triton?
OpenAI's Python-based GPU kernel language. You write kernels in Python with tile-level operations; Triton compiles them to PTX. Much easier than raw CUDA while still approaching cuBLAS-level performance.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

### CUDA vs Triton mental model
```
CUDA                          Triton
─────────────────────────     ────────────────────────────────
Thread-level programming      Block/tile-level programming
You manage shared memory      Triton manages shared memory
You write warps/syncs         Triton handles warp scheduling
Max control                   ~90% perf with ~30% the code
```

### When to use Triton vs raw CUDA
- **Triton:** new custom ops, fused attention, fast iteration
- **Raw CUDA:** max performance, existing codebases, fine control over warp/register layout

---

## 9. PyTorch Extensions (CUDA)

### Why write a PyTorch CUDA extension?
- Fuse multiple operations (e.g., layernorm + dropout) → fewer kernel launches, better cache use
- Implement ops not in PyTorch (custom attention variants, sparse ops)
- Speed up bottleneck ops identified by profiling

### Extension structure
```
my_op/
├── my_op.cu          ← CUDA kernel
├── my_op_bind.cpp    ← pybind11 bindings
└── setup.py          ← builds the extension
```

```cpp
// my_op.cu
__global__ void relu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0 ? x[i] : 0;
}

// my_op_bind.cpp
#include <torch/extension.h>
torch::Tensor relu_cuda(torch::Tensor x) {
    int n = x.numel();
    relu_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), n);
    return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu", &relu_cuda); }
```

```python
# Python usage
import my_op
out = my_op.relu(tensor.cuda())
```

### Build and load
```python
from torch.utils.cpp_extension import load
my_op = load(name="my_op", sources=["my_op.cu", "my_op_bind.cpp"], verbose=True)
```

### Profiling — find what to optimize
```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 10. Final Project — MNIST MLP

### Goal
Train a multi-layer perceptron on MNIST entirely in CUDA C (no PyTorch) to solidify everything learned.

### Architecture
```
Input (784)  →  Hidden (256, ReLU)  →  Hidden (128, ReLU)  →  Output (10, Softmax)
```

### Components you implement in CUDA
| Component | CUDA technique |
|-----------|---------------|
| Forward pass linear layer | cuBLAS SGEMM or custom tiled kernel |
| ReLU activation | Simple element-wise kernel |
| Softmax | Parallel reduction (max, then exp, then sum, then divide) |
| Cross-entropy loss | Reduction kernel |
| Backward pass (gradients) | Transposed GEMM + element-wise |
| SGD weight update | Element-wise kernel |

### Training loop (host side)
```
for each epoch:
    for each mini-batch:
        cudaMemcpy batch → device
        forward_pass()          ← chain of kernels
        compute_loss()
        backward_pass()         ← gradient kernels
        update_weights()        ← SGD kernel
    cudaMemcpy results → host
    print accuracy
```

---

## 11. Next Steps

### Recommended paths after this course
1. **GPU MODE** — Discord community for CUDA/ML kernels (GPUMODE Discord)
2. **Andrej Karpathy's llm.c** — transformer training from scratch in CUDA, builds directly on this course's skills
3. **Flash Attention** — Tri Dao's memory-efficient attention kernel, exemplifies tiling + register reuse at expert level
4. **Cutlass** — NVIDIA's production GEMM template library
5. **Nsight Compute / nvprof** — GPU profiling deep-dive

### Mental model for perf tuning
```
Profile first (nsight compute / torch profiler)
         │
         ▼
Is the kernel memory-bandwidth bound or compute bound?
         │
   BW bound ──────────────────────────────► Improve coalescing, use shared mem,
         │                                   fuse kernels, increase arithmetic intensity
         │
Compute bound ───────────────────────────► Increase occupancy, use tensor cores,
                                            reduce register pressure, use FP16
```

---

## Key Diagrams Summary

### 1. Full GPU Software Stack
```
User Python code
      │  import torch; torch.matmul(A, B)
      ▼
PyTorch dispatcher
      │  selects cuBLAS or custom kernel
      ▼
CUDA Runtime (cudart)
      │  manages context, memory, streams
      ▼
CUDA Driver (host-injected libcuda.so)
      │  translates to GPU commands
      ▼
GPU Hardware (SMs, HBM)
```

### 2. Kernel Launch → Execution Flow
```
CPU: kernel<<<G, B>>>(args)
      │
      ▼  inserted into GPU command queue
GPU: Warp Scheduler picks up thread blocks
      │
      ├─ Block → assigned to SM
      │    └─ split into Warps of 32 threads
      │         └─ SIMT execution on CUDA cores
      │
      ▼ (while waiting for memory...)
SM hides latency by switching to another warp
```

### 3. Memory Access Pattern Impact
```
Coalesced (fast):
  Thread: 0  1  2  3  4  5  6  7
  Addr:   0  1  2  3  4  5  6  7  ← 1 memory transaction

Strided (slow):
  Thread: 0  1  2  3  4  5  6  7
  Addr:   0  8  16 24 32 40 48 56  ← 8 memory transactions

Random (worst):
  Thread: 0  1  2  3  4  5  6  7
  Addr:   7  2  15 3  9  1  22 6   ← 8 memory transactions + no cache reuse
```

### 4. Tiled Matrix Multiplication
```
A (M×K)            B (K×N)
┌───────────┐      ┌─────┬─────┐
│           │      │     │     │
│   [tile]──┼──────┼►[tile]    │
│           │      │     │     │
└───────────┘      └─────┴─────┘
                         │
                         ▼
                   C (M×N)
                   ┌─────┬─────┐
                   │     │[out]│  ← one block computes one TILE×TILE of C
                   └─────┴─────┘

Each tile pair loaded ONCE into shared memory,
then reused TILE times → TILE× fewer global reads
```

---

*Sources: [freeCodeCamp article](https://www.freecodecamp.org/news/learn-cuda-programming/) · [GitHub repo](https://github.com/Infatoshi/cuda-course) · [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)*
