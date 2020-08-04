## Cuda Study Repo 

#### Streaming Multiprocessor (SM) 
CUDA GPUs are based on scalable arrays of multithreaded SMs. Each SM consists of several CUDA cores. 
Low clock rate and small cache. One SM can run multiple thread blocks in parallel and has schedulers to issue instructions to warps. It has a large amount of registers to store data from thousands of threads running at the same time. 
All warps of a thread block are on the SM until each one has finished computation. 

#### Blocks
Blocks consist of several threads (up to 1024), they are divided in to warps. These warps are assigned to SMs for execution. The scheduling is very architecture (Volta, Kepler, Turing,...) specific. 

#### Grids
Several blocks can be used to form a grid, all the blocks inside a grid have the same number of threads. Grids can be used for computations that require a large amount of parallely running thread blocks. 

#### Warps
Basic CUDA execution unit. A warp consists of 32 threads, threads inside a warp can be assumed to execute in lockstep. In order to synch different warps, we can use : 
```C++
__syncthreads(); 
```
#### Access Patterns
Coalesced global memory access is desired to avoid cache misses and use cache lines effeciently. 
Structures of Arrays (SOA) is preferred for SIMT models like CUDA, while CPUs perefer sequential AOS for cache efficiency.

#### Shared Memory 
Is user managed cache, used to read from global memory and store it in a coalesced way. Only visible to threads within the same block. It has similiar benefits to CPU cache, but unlike CPU cache it can be user managed. Shared memory has lower latency than global memory and higher bandwith than global memory. 

#### Banks
For higher bandwith, shared memory is organized in banks. Each bank can service one address per cycle, that means that multiple accesses by threads in the same warp to a bank results in a so called bank conflict. The worst case is a 32-way conflict with 31 replays, each replay adds a few cycles of latency. 

#### Read-only data/cache (or texture cache/memory) 
A read-only cache is suitable for storing data that is read-only and does not change the course of kernel execution. It is optimized for this and therefore reduces load on the other caches. It's data is visible to all threads in a grid. The memory is read-only for GPU, however CPU can read and write from it. Objects marked 
```C++
const __restrict__
```
are read only. Loading through this cache can be forced with :
```C++
__ldg
```
Best used when one warp reads from same address/data (broadcast of all of the threads). Otpimized for 2D/3D locality. Optimal for Image Processing for example, supports bilinear/trilinear interpolation. 

#### Registers 

GPUs have lots of registers compared to CPU, this reduces the latency of context switching. Every thread can access only it's own regsisters, local variables declared inside the scope are stored inside the registers. The compiler finds the best # of registers per thread during compilation. 

#### Register Spills 
 
Happens when too many registers are declared and therefore data is moved to L1/L2 cache or even global memory. To avoid this the programmmer should make sure to limit the number of local variables that are declared. This can be done by splitting one complex kernel in a few simpler ones. 

#### Pinned memory 

Data moves from host to device memory (PCIE) or device to device memory (NVLink). In order to avoid getting chocked on the bus, it is recommended to : 

1) minimize the amount of data that is transfered, might even mean to run code on GPU sequentially in order to avoid extra transfer
2) higher banwith through pinned memory 
3) batch small transfers into one large transfer, reduces transfer data CUDA API latency. (ranges from µs to ms, based on System Config)
4) Asynchronous transfer to overlap execution with data transfers. 

#### Pinned vs pageable memory 

GPU will not access pageable memory, CUDA driver copies data from pageable into pinned memory, transfers to device via DMA (Device Memory Controller) => extra latency 
=> has a chance to move requested page to GPU memory, which has been swapped and brought back to GPU. 

```C++
cudaMallocHost(); 
``` 
=> makes memory pinned 
 
Pinned memory bandwith is higher for low data sizes, pageable bandwith is higher with large data sizes. (Due to techniques such as overlap that are used by the DMA engine)
Allocating the whole mem as pinned, can reduce overall system performance as it takes away pages for other tasks. There is no right formula for the amount of pinned memory that should be applied, it is depending on the system heavily.  

#### Unified memory  

Accessible by all CPUs and GPUs on the node. Should be accessed in a coalesced way.  Uses ```cudaMallocManaged()``` instead of ```malloc()```
The variables only need to be declared once, unlike what we previously saw => simpler for programmers. 
```cudaMallocManaged()``` delcares memory on a first touch basis, so if the mem is first allocated on CPU, the page will be mapped to the CPU. So if we acces it from GPU, there will be a page fault and the time will of page migration will be additionall overhead that occurs in this case. 


#### Page migration 

1) allocate new pages on GPU and CPU (first touch basis, see previous section). If it is not present, a page table fault happens. 
2) Old page is unmapped 
3) data is copied from CPU to GPU 
4) new pages are mapped on the GPU, old pages are freed on the CPU 

#### Translation Lookaside Buffer (TLB)

Like in CPU, maps physical to virtual address. TLB is locked when TLB occurs, new instructions will be stalled, until preceding steps were performed. => necessary to maintain coherency and maintain memory state in the SM. Responsible for removing duplicates, updating the mapping and transferring the page data. All of this time is added to the kernel time. 

How to solve this ? : 
1) create init kernel on GPU, so that there are no page faults, then optimize the page faults by using the warp per page control concept
2) prefetch the data 

#### Optimizing unified memory with warp per page control idea 

Adding a kernel to init the array in the GPU itself => pages are allocated and mapped to the GPU memory (first touch).
=> we can see that unified_initialzed.cu (Chapter02/ Folder 4) performs orders of magnitude better on bandwith than the naive version. 

No host to device row now, however init takes the longest now . To see indiviudal page faults we can use : 
```sh 
nvprof --print-gpu-trace
``` 

In the unified_initialzed.cu case, 11 page faults occur. When access is complicated, the driver does not have enough information about to migrate to GPU. Therefore warp per page is used. 


#### Warp per Page 

Means that each warp will access memory that is in the same page : 
```C++
#define STRIDE_64K 65536

__global__ void init(int n, float *x, float *y) {
    int lane_id = threadIdx.x & 31; //31 because one warp consists of 32 threads 
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5; 
    size_t warps_per_grid = (blockDim.x * gridDim.x) >> 5; 
    size_t warp_total = ((sizeof(float)*n) + STRIDE_64K-1)/STRIDE_64K; 
    for(;warp_id < warp_total;warp_id += warps_per_grid)
    {
        //pragma unroll
        for(int rep = 0; rep < STRIDE_64K/sizeof(float)/32; ++rep) {
            size_t ind = warp_id * STRIDE_64K/sizeof(float)/32 + rep * 32 + lane_id; 
            if(ind < n) {
                x[ind] = 1.0f; 
                y[ind] = 2.0f; 
            }
        }
    } 
}
```
Each warp manages 64 KB.  


#### Data prefetching 

Hints to the driver to prefetch data that might be used next, the API is called 
```C++
cudaMemPrefetchAsync(); 
``` 

if we know what memory will be used on which device, we can prefetch it. 
With unified memory, the limited GPU memory can be extended. 
You can get higher performance with unified memory, with prefetching and hints were data is located (cudaMemAdvice()) can help in multi processor cases a lot. 

