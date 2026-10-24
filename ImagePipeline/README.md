# ImagePipeline (CUDA C++17, Windows / Visual Studio 2022)

Prerequisites: Visual Studio 2022 (Desktop C++), CUDA Toolkit 11.8+ or 12.x, CMake 3.20+ (bundled with VS).

**Option A – open folder in VS 2022:** *File → Open → Folder…*, select this directory; VS configures CMake. Choose `ImagePipeline.exe` and build (x64-Release).
**Option B – generate a .sln:**
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86   # your GPU's compute capability
cmake --build build --config Release
build\ImagePipeline.sln   # open in VS
```
Run tests: `ctest --test-dir build -C Release --output-on-failure`
Run: `build\Release\ImagePipeline.exe <input_dir> <output_dir> [--threshold 20] [--sigma 1.0] [--min-area 16] [--slots 3]`
Test data: `python tools\make_test_images.py data\in 24`
See DESIGN.md for architecture, optimisations and the (to be filled) results table.
