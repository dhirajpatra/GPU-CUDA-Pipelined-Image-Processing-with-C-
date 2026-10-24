# GPU-Pipelined Image Processing – Design Overview

Language/toolchain: **C++17 + CUDA** (MSVC 2022, CUDA Toolkit 11.8+ / 12.x), Windows x64.

## 1. Requirement translation

| Requirement | Decision |
|---|---|
| BMP input, differing resolutions | Own BMP decoder (8/24/32-bit, bottom-up/top-down, row padding). Headers are scanned first; every buffer is sized once for the largest image. |
| 4 concurrent stages | Four stages, each with its own CPU thread **and** CUDA stream, linked by bounded queues and CUDA events. |
| Gaussian 3x3, sigma 1 | Normalised kernel in constant memory; luma conversion fused into the tile load. |
| Laplacian edge detection | 4-neighbour kernel `[0 1 0; 1 -4 1; 0 1 0]`, `abs`, clamped to 255. |
| Threshold + colour overlay | Edge pixel if `|response| >= threshold`; each object painted in its own colour on the *original* image; white cross on the centroid. |
| Centroids + object count | GPU connected-component labeling (8-connectivity) with per-object statistics. |
| Metrics per stage | CUDA events around every stage; report of avg time, total time, GPU-busy % of wall time, throughput, latency, stage balance. |

**Ambiguities and the assumption taken (all configurable or documented):**
- *"Object"* = an 8-connected set of edge pixels with at least `--min-area` pixels (default 16); smaller sets are noise. The centroid is the mean of the object's edge-pixel coordinates (its contour), which for closed shapes coincides with the shape centre.
- Laplacian response is taken as `abs` so both sides of an edge pass the same threshold. With a 3x3 Gaussian in front, an ideal step becomes an edge band ~4 px wide; the default threshold (20) is low enough that the band is connected (the unit tests derive this analytically).
- Image borders: clamp-to-edge for both filters (no shrinking of the image, no dark frame).
- Gray = BT.601 luma, rounded. Gaussian output is rounded to 8 bit (matches the worked example in the brief).
- Input folder may contain non-BMP or corrupt files: they are skipped with a warning.

## 2. Architecture

```
 CPU thread          CPU thread        CPU thread       CPU thread        CPU thread
┌───────────┐       ┌──────────┐      ┌──────────┐    ┌──────────────┐   ┌──────────┐
│ Loader    │──q──▶ │ Gaussian │─q──▶ │Laplacian │─q─▶│ Threshold +  │─q▶│ Writer   │
│ BMP→pinned│       │ stream 1 │      │ stream 2 │    │ label+overlay│   │ BMP+CSV  │
│ async H2D │       └──────────┘      └──────────┘    │ stream 3 +D2H│   └────┬─────┘
└─────▲─────┘                                         └──────────────┘        │
      └───────────────────── free-frame pool (N slots) ◀─────────────────────┘
```

* **Frame pool (`Frame`)**: N slots (default 3), each owning pinned host in/out buffers and device BGR/gray/edge buffers plus CUDA events. The pool *is* the back-pressure mechanism: the loader blocks when all slots are in flight, so memory is bounded and no stage can run away.
* **Stage hand-off without CPU stalls**: stage *k* records `stop[k]` on its stream; stage *k+1* calls `cudaStreamWaitEvent` on it. Dependencies are resolved on the GPU; the CPU threads only enqueue work. While frame *i* is in the Laplacian kernel, frame *i+1* is in the Gaussian kernel and frame *i+2* is being copied to the GPU.
* **Design patterns**: pipeline / pipes-and-filters (stages + bounded `BlockingQueue`), object pool (`Frame`), RAII for every CUDA resource (`DeviceBuffer`, `PinnedBuffer`, `CudaStream`, `CudaEvent`), single-writer ownership (`ComponentAnalyzer` scratch is only touched by the final stage thread; CSV files only by the writer).
* **Error handling**: the first exception from any thread is stored, all queues are aborted (waking every blocked thread) and the exception is re-thrown from `Pipeline::run()` after joining.
* **Modules**: `Bmp` (I/O) · `Kernels` (device code + launchers + `ComponentAnalyzer`) · `Frame`/`Pipeline` (orchestration) · `Stats` (reporting) · `main` (CLI). Adding a stage = add a kernel launcher, one queue and one `computeStage(...)` call; stage ids/names live in `Stats.h`.

## 3. GPU and memory optimisations

1. **Pinned host memory** + `cudaMemcpyAsync` on a dedicated stream → H2D/D2H overlap with kernels; only the pixels cross PCIe (3 B/px in, 3 B/px out). Intermediate images (gray, blur, edge, labels) never leave the GPU. The object list is a fixed 48 KB copy.
2. **No allocation in steady state**: all buffers pre-sized from the header scan (`cudaMalloc/Free` implicitly serialise the device).
3. **Shared-memory tiling** (32x8 block + 1-pixel halo) for both filters: each input byte is read from global memory ~1.3x instead of 9x / 5x; coalesced 32-byte-wide rows. Luma conversion is fused into the Gaussian load, saving a whole kernel and a full-image round trip.
4. **8-bit intermediates** (gray, edge): 4x less traffic than float images; the stages are bandwidth-bound.
5. **Connected components** by label equivalence: union–find forest, `atomicMin` hooking of the larger root under the smaller one, pointer jumping each pass; host checks convergence only every 2 passes (1 int copied). Statistics (`count, Σx, Σy`) via atomics on the root; only root pixels emit a `Detection`.
6. **Non-blocking streams**; events instead of `cudaDeviceSynchronize`; the only host synchronisation inside the pipeline is in the final stage (it needs the object count) and at the end of each frame.
7. Memory per slot: 3 B/px (host in) + 3 B/px (host out) + 5 B/px device; shared analyzer scratch 24 B/px. A 3840x2160 image with 3 slots ≈ 0.45 GB total.

### Known limitations / next optimisation steps
- The final stage is intrinsically the heaviest (labeling needs several passes and host round trips) and dominates the pipeline period; the filters are cheap. See §5.
- Atomics on a very large single object contend. Warp-aggregated accumulation (`__match_any_sync`) or per-block shared-memory reduction would remove this.
- A block-local labeling pass (tile-level union–find in shared memory) followed by a border merge would cut the number of global passes.
- Pixel indices are `int` → images up to ~700 Mpixel (enforced by the BMP reader).

## 4. Stage-time balance (requirement: "approximately the same duration")
Filter kernels are bandwidth-bound and take roughly the same time per frame; labeling+overlay is several times longer. The pipeline period is set by the slowest stage, so the report prints `balance = slowest/fastest`. Ways to even it out, in order of preference: (a) make the final stage cheaper (see limitations above), (b) split it into two pipeline stages (labeling | statistics+overlay) – trivial with `computeStage`, (c) move the Laplacian and Gaussian into one stage if they are much faster than the rest, to avoid idle streams. Note that this is a property of the workload, not hidden: the `metrics.csv` makes it visible on any GPU.

## 5. Benchmarking and profiling

```
python tools\make_test_images.py data\in 24          # mixed resolutions 333x217 … 3840x2160
ImagePipeline.exe data\in data\out                   # pipelined (slots=3)
ImagePipeline.exe data\in data\out_serial --slots 1  # serial baseline for comparison
```
Outputs: `<name>_edges.bmp`, `summary.csv` (count per image), `centroids.csv`, `metrics.csv`, plus the console report (per-stage avg ms, GPU-busy % of wall time, throughput, latency, balance). For kernel-level detail (occupancy, achieved bandwidth, idle gaps between kernels) use **Nsight Systems** (`nsys profile ImagePipeline.exe …`, the three streams should show overlapping rows) and **Nsight Compute** on `gaussianKernel` / `laplacianKernel`.

### Measured results
> **Not included in this document.** The package was authored and functionally verified in an environment without a GPU or Windows, so no performance numbers are claimed. Run the commands above and paste `metrics.csv` here:

| Stage | avg ms/frame | GPU busy % of wall |
|---|---|---|
| H2D copy | | |
| Gaussian | | |
| Laplacian | | |
| Threshold+label+overlay | | |
| D2H copy | | |
| **Throughput (fps)** pipelined vs `--slots 1` | | |

## 6. Coding standards
Namespace `ipp`; `PascalCase` types, `camelCase` functions/variables, `kConstant`, `member_` suffix for private data, `c_` prefix for `__constant__`. Doc comment on every public type/function and on every kernel (what it computes, thread mapping). RAII only – no naked `cudaMalloc`/`new`. All CUDA calls wrapped by `IPP_CUDA_CHECK` (file/line in the message). Warnings at `/W4`. No global mutable state except the Gaussian constant-memory table (set once before the pipeline starts).

## 7. Unit-test strategy (`ImagePipelineTests`, run via `ctest` or directly)
| Area | Tests |
|---|---|
| BMP | round trip for widths 1–33 (all padding cases), 8-bit palette + bottom-up, garbage/truncated files rejected |
| Queue | producer/consumer sum, drain-after-close, push-after-close |
| Gaussian | kernel normalised/symmetric; constant image preserved (odd size); equals CPU reference on 61x45 random image (±1 LSB for fma); 1x1 image |
| Laplacian | constant → 0; vertical step has known values; **bit-exact** vs CPU reference on 70x33 random image |
| Components | two squares → 2 objects, exact centroids/areas, overlay painted only on edges, centroid cross; min-area noise filter; 8-connectivity diagonal; 96x96 spiral (long dependency chain → single object, exact area); empty image; object at image border |
| Pipeline | 3 images of different sizes + 1 corrupt file → exact counts, corrupt skipped, output sizes match, **`--slots 3` output identical to `--slots 1`**, empty directory throws |

GPU tests are skipped (not failed) on machines without a CUDA device. Performance is verified through the benchmark report, not unit tests, because wall-time assertions are flaky.

**Verification status:** the full suite (19 tests) passes when the real sources are compiled against a CPU emulation of the CUDA runtime (kernels executed thread-by-thread, real multi-threaded pipeline). That validates algorithms, indexing, hand-offs and shutdown logic; it does not exercise real GPU timing/race behaviour, so run the suite on your GPU first.
