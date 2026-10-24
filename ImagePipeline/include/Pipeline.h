// Pipeline.h - the concurrent, GPU-based image-processing pipeline.
#pragma once

#include <cstdint>
#include <filesystem>

#include "Stats.h"

namespace ipp {

class Pipeline {
public:
    struct Config {
        std::filesystem::path inputDir;
        std::filesystem::path outputDir;
        std::uint8_t threshold = 20;   ///< Edge pixel if |Laplacian| >= threshold.
        float sigma = 1.0f;            ///< Gaussian sigma (3x3 kernel).
        std::uint32_t minArea = 16;    ///< Smallest component (in edge pixels) reported as an object.
        int slots = 3;                 ///< Frames in flight. 1 = fully serial baseline.
        int device = 0;                ///< CUDA device ordinal.
    };

    explicit Pipeline(Config config);

    /// Processes every *.bmp in inputDir. Writes, into outputDir:
    ///   <name>_edges.bmp, summary.csv, centroids.csv, metrics.csv
    /// Throws on fatal errors (no GPU, unreadable directory, CUDA failure).
    PipelineReport run();

private:
    Config config_;
};

}  // namespace ipp
