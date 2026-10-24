#include "Bmp.h"

#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ipp {
namespace {

namespace fs = std::filesystem;

// Largest image we accept: keeps every pixel index (including *3 for BGR) inside int32.
constexpr std::uint64_t kMaxPixels = static_cast<std::uint64_t>(INT_MAX) / 3u;

struct BmpHeader {
    std::int32_t width = 0;
    std::int32_t height = 0;  // always positive; orientation is in topDown
    std::uint16_t bitsPerPixel = 0;
    std::uint32_t compression = 0;
    std::uint32_t colorsUsed = 0;
    std::uint32_t dataOffset = 0;
    std::uint32_t dibSize = 0;
    bool topDown = false;
};

// BMP is little-endian; every supported target (x86-64, ARM64 Windows) is too.
template <typename T>
T readLE(const std::uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

template <typename T>
void putLE(std::uint8_t* p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

BmpHeader parseHeader(std::ifstream& in, const fs::path& path) {
    const std::string name = path.string();
    std::uint8_t b[54];
    in.read(reinterpret_cast<char*>(b), sizeof(b));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(b))) {
        throw std::runtime_error(name + ": file too small to be a BMP");
    }
    if (b[0] != 'B' || b[1] != 'M') throw std::runtime_error(name + ": missing 'BM' signature");

    BmpHeader h;
    h.dataOffset = readLE<std::uint32_t>(b + 10);
    h.dibSize = readLE<std::uint32_t>(b + 14);
    if (h.dibSize < 40) throw std::runtime_error(name + ": unsupported DIB header (need BITMAPINFOHEADER or newer)");

    h.width = readLE<std::int32_t>(b + 18);
    const std::int32_t rawHeight = readLE<std::int32_t>(b + 22);
    h.bitsPerPixel = readLE<std::uint16_t>(b + 28);
    h.compression = readLE<std::uint32_t>(b + 30);
    h.colorsUsed = readLE<std::uint32_t>(b + 46);

    if (rawHeight == INT_MIN) throw std::runtime_error(name + ": invalid height");
    h.topDown = rawHeight < 0;
    h.height = h.topDown ? -rawHeight : rawHeight;

    if (h.width <= 0 || h.height <= 0) throw std::runtime_error(name + ": non-positive dimensions");
    if (static_cast<std::uint64_t>(h.width) * static_cast<std::uint64_t>(h.height) > kMaxPixels) {
        throw std::runtime_error(name + ": image too large");
    }
    const bool bppOk = h.bitsPerPixel == 8 || h.bitsPerPixel == 24 || h.bitsPerPixel == 32;
    if (!bppOk) throw std::runtime_error(name + ": unsupported bit depth " + std::to_string(h.bitsPerPixel));
    // 0 = BI_RGB; 3 = BI_BITFIELDS is accepted for 32-bit only (assumed standard BGRA masks).
    if (!(h.compression == 0 || (h.compression == 3 && h.bitsPerPixel == 32))) {
        throw std::runtime_error(name + ": compressed BMPs are not supported");
    }
    return h;
}

}  // namespace

void readBmpDimensions(const fs::path& path, int& width, int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path.string() + ": cannot open file");
    const BmpHeader h = parseHeader(in, path);
    width = h.width;
    height = h.height;
}

void readBmpInto(const fs::path& path, std::uint8_t* destination, std::size_t capacityBytes, int& width,
                 int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path.string() + ": cannot open file");
    const BmpHeader h = parseHeader(in, path);

    const std::size_t w = static_cast<std::size_t>(h.width);
    const std::size_t rows = static_cast<std::size_t>(h.height);
    if (capacityBytes < w * rows * 3) throw std::runtime_error(path.string() + ": destination buffer too small");

    std::vector<std::uint8_t> palette;  // BGRX entries
    std::size_t paletteEntries = 0;
    if (h.bitsPerPixel == 8) {
        paletteEntries = h.colorsUsed != 0 ? h.colorsUsed : 256;
        if (paletteEntries > 256) throw std::runtime_error(path.string() + ": invalid palette size");
        palette.resize(paletteEntries * 4);
        in.seekg(static_cast<std::streamoff>(14u + h.dibSize));
        in.read(reinterpret_cast<char*>(palette.data()), static_cast<std::streamsize>(palette.size()));
        if (in.gcount() != static_cast<std::streamsize>(palette.size())) {
            throw std::runtime_error(path.string() + ": truncated palette");
        }
    }

    const std::size_t rowBytes = ((w * h.bitsPerPixel + 31) / 32) * 4;  // rows are 4-byte aligned
    std::vector<std::uint8_t> row(rowBytes);
    in.seekg(static_cast<std::streamoff>(h.dataOffset));

    for (std::size_t r = 0; r < rows; ++r) {
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(rowBytes));
        if (in.gcount() != static_cast<std::streamsize>(rowBytes)) {
            throw std::runtime_error(path.string() + ": truncated pixel data");
        }
        const std::size_t y = h.topDown ? r : rows - 1 - r;  // normalise to top-down
        std::uint8_t* out = destination + y * w * 3;
        switch (h.bitsPerPixel) {
            case 24:
                std::memcpy(out, row.data(), w * 3);
                break;
            case 32:
                for (std::size_t x = 0; x < w; ++x) std::memcpy(out + x * 3, row.data() + x * 4, 3);  // drop alpha
                break;
            default:  // 8-bit palettised
                for (std::size_t x = 0; x < w; ++x) {
                    const std::size_t idx = row[x];
                    if (idx < paletteEntries) {
                        std::memcpy(out + x * 3, palette.data() + idx * 4, 3);
                    } else {
                        std::memset(out + x * 3, 0, 3);
                    }
                }
                break;
        }
    }
    width = h.width;
    height = h.height;
}

Image readBmp(const fs::path& path) {
    Image img;
    readBmpDimensions(path, img.width, img.height);
    img.bgr.resize(static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height) * 3);
    readBmpInto(path, img.bgr.data(), img.bgr.size(), img.width, img.height);
    return img;
}

void writeBmp(const fs::path& path, const std::uint8_t* bgr, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("writeBmp: non-positive dimensions");
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t rows = static_cast<std::size_t>(height);
    const std::size_t rowBytes = ((w * 3 + 3) / 4) * 4;
    const std::uint64_t imageSize = static_cast<std::uint64_t>(rowBytes) * rows;
    if (imageSize + 54 > 0xFFFFFFFFull) throw std::invalid_argument("writeBmp: image too large for BMP");

    std::uint8_t header[54] = {};
    header[0] = 'B';
    header[1] = 'M';
    putLE<std::uint32_t>(header + 2, static_cast<std::uint32_t>(imageSize + 54));
    putLE<std::uint32_t>(header + 10, 54);
    putLE<std::uint32_t>(header + 14, 40);
    putLE<std::int32_t>(header + 18, width);
    putLE<std::int32_t>(header + 22, height);  // positive = bottom-up
    putLE<std::uint16_t>(header + 26, 1);
    putLE<std::uint16_t>(header + 28, 24);
    putLE<std::uint32_t>(header + 34, static_cast<std::uint32_t>(imageSize));
    putLE<std::int32_t>(header + 38, 2835);
    putLE<std::int32_t>(header + 42, 2835);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(path.string() + ": cannot create file");
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    const std::uint8_t pad[3] = {0, 0, 0};
    const std::size_t padBytes = rowBytes - w * 3;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* src = bgr + (rows - 1 - r) * w * 3;
        out.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(w * 3));
        out.write(reinterpret_cast<const char*>(pad), static_cast<std::streamsize>(padBytes));
    }
    if (!out) throw std::runtime_error(path.string() + ": write failed");
}

}  // namespace ipp
