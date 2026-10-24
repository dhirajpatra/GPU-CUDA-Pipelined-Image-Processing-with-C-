# ImagePipeline

A GPU-based, four-stage pipelined image-processing system that detects and visualises objects in a stream of BMP images.

**Language:** C++17 (host code) + CUDA C++17 (GPU kernels in `src/Kernels.cu`)
**Platform:** Windows x64, Visual Studio 2019/2022, NVIDIA GPU (compute capability 7.5+ by default)

## Pipeline

```
Load BMP -> Gaussian 3x3 -> Laplacian edge -> Threshold + label + overlay -> Write results
(CPU+H2D)    (GPU)           (GPU)             (GPU, then D2H copy)          (CPU)
```

All stages run concurrently on separate CUDA streams. While one image is being filtered, the next is being loaded and copied to the GPU.

For each image the program produces:
- `<name>_edges.bmp`: original image with each detected object's edges coloured and a white cross on its centroid
- `summary.csv`: object count per image
- `centroids.csv`: centroid coordinates and edge-pixel count per object
- `metrics.csv`: per-stage timing and GPU utilisation

## Requirements

- Visual Studio 2022 (Desktop development with C++)
- NVIDIA CUDA Toolkit 11.8 or newer (12.x recommended)
- CMake 3.20+ (bundled with Visual Studio)
- Python 3 with `numpy` (only for the test-image generator)

## Build

**Option A: open the folder in Visual Studio**
1. File -> Open -> Folder... and select this directory.
2. Wait for CMake to configure, choose the `x64-Release` configuration.
3. Build `ImagePipeline.exe`.

**Option B: generate a solution**
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```
Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability (e.g. 75 Turing, 86 Ampere, 89 Ada). The generated `build\ImagePipeline.sln` opens in Visual Studio.

## Run

```
ImagePipeline.exe <input_dir> <output_dir> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--threshold N` | 20 | Edge pixel if abs(Laplacian) >= N (0-255) |
| `--sigma S` | 1.0 | Gaussian sigma (3x3 kernel) |
| `--min-area N` | 16 | Smallest object, in edge pixels; smaller sets are treated as noise |
| `--slots N` | 3 | Frames in flight; `1` gives a fully serial baseline |
| `--device N` | 0 | CUDA device index |

Generate a mixed-resolution test set and run:
```
python tools\make_test_images.py data\in 24
ImagePipeline.exe data\in data\out
ImagePipeline.exe data\in data\out_serial --slots 1
```
Comparing the two runs shows the benefit of pipelining. Only `.bmp` files (8, 24 or 32-bit, uncompressed) are read; invalid files are skipped with a warning.

## Tests

```
ctest --test-dir build -C Release --output-on-failure
```
or run `ImagePipelineTests.exe` directly. The suite covers BMP I/O, both filters against CPU references, object detection edge cases, and an end-to-end run (a serial run and a pipelined run must give identical output). GPU tests are skipped if no CUDA device is found.

## Profiling

For kernel-level detail use NVIDIA Nsight Systems (`nsys profile ImagePipeline.exe ...`; the streams should show overlapping work) and Nsight Compute on `gaussianKernel` / `laplacianKernel`.

## Project layout

```
include/   CudaUtils.h (RAII wrappers)  BlockingQueue.h  Bmp.h  Kernels.h
           Frame.h  Pipeline.h  Stats.h
src/       Bmp.cpp  Kernels.cu  Pipeline.cpp  main.cpp
tests/     Tests.cpp
tools/     make_test_images.py
DESIGN.md  Architecture, optimisations, assumptions, test strategy
```

## Notes

- See `DESIGN.md` for the design rationale, assumptions about the brief, known limitations and the results table to fill in.
- The code was functionally verified against a CPU emulation of CUDA; performance has not been measured on a real GPU yet.
