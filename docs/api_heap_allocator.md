# Heap Arena Allocator (allocator.cuh)

`DeviceHeap` is a large-object heap embedded directly in `DevicePool`. Design:

- **`HEAP_NUM_ARENAS` arenas** (default 64, overridable via `COACD_GPU_ARENAS=N pip install -e .`), selected by `blockIdx.x % HEAP_NUM_ARENAS`. Each arena has **64 sub-bin doubly-linked free lists**, a **64-bit occupancy bitmap**, and a spin-lock.
- **Sub-bins**: 32 pow-of-2 bins x 2 linear halves. Bin b = `[512*2^b, 1024*2^b)`. Sub-bin 2b = lower half `[512*2^b, 768*2^b)`, sub-bin 2b+1 = upper half `[768*2^b, 1024*2^b)`. Minimum alignment: `HEAP_ALIGN = 512` bytes.
- **Block layout**: `[HeapBlockHdr (16 B)][data (data_size B)][HeapBlockFtr (16 B)]`. `HeapBlockHdr`/`Ftr` store `data_size`, `arena_idx`, `is_free`. Free blocks store `prev`/`next` pointers in first 16 bytes of data.
- **Slab layout**: each new pool slab has `[leading sentinel 32B][free block][trailing sentinel 32B]`. Sentinels (`is_free=0`, `data_size=0`) prevent coalescing across slab boundaries.
- **Allocation** (thread 0 only): bitmap search for lowest eligible sub-bin (O(1) via `__ffsll`); pop head; split remainder if >= `HEAP_HDR_SIZE + HEAP_ALIGN + HEAP_FTR_SIZE`. If no sub-bin has a free block, allocate a new slab from the pool (min 128 KB or next-pow-2).
- **Free** (thread 0 only): inspect prev footer and next header; coalesce adjacent free blocks via doubly-linked O(1) removal; insert merged block into correct sub-bin. Arena for insertion = block's stored `arena_idx` (all blocks in a slab share the same arena, so coalescing is always intra-arena).
- **No compact**: `beam_heap_compact()` is a no-op -- coalescing is handled in-place by `heap_free`. Pool remains stable across all calls.

## DevicePool layout (must match csrc/structs.h)

```
DevicePool { base, offset*, capacity, DeviceHeap heap, DeviceHeap scratch }
DeviceHeap { DevicePool* pool (back-ptr set by heap_init_kernel), HeapArena arenas[HEAP_NUM_ARENAS] }
HeapArena  { bitmap, lock, _pad, heads[64], tails[64] }  // 1040 bytes each
```
`sizeof(DevicePool)` ~ 133 KB at default HEAP_NUM_ARENAS=64 (bulk is the two embedded heap arrays).

Host init: allocate one `d_pool_struct` of `sizeof(struct DevicePool)`, set `base`/`offset`/`capacity`, then launch `heap_init_kernel<<<2*HEAP_NUM_ARENAS,32>>>` which sets `heap.pool = scratch.pool = pool` and zeroes all arena state.

## Benchmark results

(100 hulls x 2000 pts/hull, gaussian, HEAP_NUM_ARENAS=64):
- **Pool stable at ~172 MB** after first call regardless of compact frequency (compact is a no-op).
- Arena count sweep (`docs/arena_sweep.md`): pool scales ~linearly with HEAP_NUM_ARENAS; throughput is unaffected. A=32 saves ~10% pool with no performance cost.
