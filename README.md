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
Colaesced global memory access is desired to avoid cache misses and use cache lines effeciently. 
Structures of Arrays (SOA) is preferred for SIMT models like CUDA, while CPUs perefer sequential AOS for cache efficiency.

#### Shared Memory 
Is user managed cache, used to read from global memory and store it in a coalesced way. Only visible to threads within the same block. It has similiar benefits to CPU cache, but unlike CPU cache it can be user managed. Shared memory has lower latency than global memory and higher bandwith than global memory. 

#### Banks
For higher bandwith, shared memory is organized in banks. Each bank can service one address per cycle, that means that multiple accesses by threads in the same warp to a bank results in a so called bank conflict. The worst case is a 32-way conflict with 31 replays, each replay adds a few cycles of latency. 

#### Read-only data/cache (or texture cache) 
A read-only cache is suitable for storing data that is read-only and does not change the course of kernel execution. It is optimized for this and therefore reduces load on the other caches. It's data is visible to all threads in a grid. The memory is read-only for GPU, however CPU can read and write from it. Objects marked 
```C++
const __restrict__
```
are read only. Loading through this cache can be forced with :
```C++
__ldg
```
Best used when one warp reads from same address/data (broadcast of all of the threads). Otpimized for 2D/3D locality. Optimal for Image Processing for example, supports bilinear/trilinear interpolation. 

##### Texture Dim : defines array dimension (1D,2D,3D) 
##### Texture type : defines texel in terms of floating point or basic int 
##### Texture Read Mode : defines read mode as ```NormalizedFloat (range 0 to 1)``` or ```ModeElement (range -1 to 1)``` 
##### Texture Addressing Mode : defines out of range addressing such as clamping, warping, mirroring 
##### Texutre Filtering Mode : how return value is computer when fetching the texture (interpolation is possible in linear mode) 



