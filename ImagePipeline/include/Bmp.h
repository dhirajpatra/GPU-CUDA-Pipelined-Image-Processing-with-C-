// Bmp.h - minimal, dependency-free BMP reader/writer.
//
// Reads uncompressed 8-bit (palettised), 24-bit and 32-bit BMPs, bottom-up or
// top-down, and always produces tightly packed top-down BGR (3 bytes/pixel).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ipp {

/// Tightly packed top-down BGR image (row stride = width * 3).
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> bgr;
};

/// Reads only the header. Throws std::runtime_error if the file is not a supported BMP.
void readBmpDimensions(const std::filesystem::path& path, int& width, int& height);

/// Decodes `path` straight into caller-provided memory (e.g. a pinned buffer).
/// Throws std::runtime_error on unsupported/corrupt files or if `capacityBytes`
/// is smaller than width*height*3.
void readBmpInto(const std::filesystem::path& path, std::uint8_t* destination,
                 std::size_t capacityBytes, int& width, int& height);

/// Convenience wrapper around readBmpInto for tests and tools.
Image readBmp(const std::filesystem::path& path);

/// Writes a 24-bit BMP from tightly packed top-down BGR data.
void writeBmp(const std::filesystem::path& path, const std::uint8_t* bgr, int width, int height);

}  // namespace ipp
