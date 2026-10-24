// Stats.h - per-stage timing statistics and report formatting.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <ostream>

namespace ipp {

/// Pipeline stages in execution order. The three "compute" stages are the
/// Gaussian, Laplacian and final stage; the copies are reported separately.
constexpr int kNumStages = 5;
enum StageId : int { kStageH2D = 0, kStageGaussian = 1, kStageEdge = 2, kStageFinal = 3, kStageD2H = 4 };
constexpr std::array<const char*, kNumStages> kStageNames = {
    "H2D copy", "Gaussian filter", "Laplacian edge", "Threshold+label+overlay", "D2H copy"};

struct StageStats {
    std::size_t frames = 0;
    double totalMs = 0.0;  ///< Sum of GPU time (CUDA events) over all frames.
    double avgMs() const { return frames ? totalMs / static_cast<double>(frames) : 0.0; }
};

struct PipelineReport {
    std::array<StageStats, kNumStages> stages{};
    std::size_t frames = 0;
    std::size_t skippedFiles = 0;
    double wallMs = 0.0;           ///< Wall-clock time of the whole run.
    double totalLatencyMs = 0.0;   ///< Sum of (load start -> output written) per frame.
    double totalReadMs = 0.0;      ///< CPU time decoding BMPs.
    double totalWriteMs = 0.0;     ///< CPU time encoding/writing outputs.

    double throughputFps() const { return wallMs > 0.0 ? 1000.0 * static_cast<double>(frames) / wallMs : 0.0; }
    double avgLatencyMs() const { return frames ? totalLatencyMs / static_cast<double>(frames) : 0.0; }
    /// Fraction of wall time the GPU spent executing work of stage `s`.
    double utilisation(int s) const { return wallMs > 0.0 ? stages[static_cast<std::size_t>(s)].totalMs / wallMs : 0.0; }
    /// Slowest / fastest compute stage (1.0 = perfectly balanced).
    double balanceRatio() const {
        double lo = 1e300, hi = 0.0;
        for (int s = kStageGaussian; s <= kStageFinal; ++s) {
            const double a = stages[static_cast<std::size_t>(s)].avgMs();
            lo = std::min(lo, a);
            hi = std::max(hi, a);
        }
        return lo > 0.0 ? hi / lo : 0.0;
    }
};

inline void printReport(const PipelineReport& r, std::ostream& os) {
    os << std::fixed << std::setprecision(3);
    os << "\n=== Pipeline performance report ===\n";
    os << "Frames processed : " << r.frames << " (skipped invalid files: " << r.skippedFiles << ")\n";
    os << "Wall time        : " << r.wallMs << " ms\n";
    os << "Throughput       : " << r.throughputFps() << " frames/s\n";
    os << "Avg latency/frame: " << r.avgLatencyMs() << " ms (load start -> output written)\n";
    os << "Avg CPU decode   : " << (r.frames ? r.totalReadMs / static_cast<double>(r.frames) : 0.0) << " ms\n";
    os << "Avg CPU encode   : " << (r.frames ? r.totalWriteMs / static_cast<double>(r.frames) : 0.0) << " ms\n\n";
    os << std::left << std::setw(26) << "Stage" << std::right << std::setw(14) << "avg ms/frame"
       << std::setw(14) << "total ms" << std::setw(16) << "GPU busy % wall\n";
    for (int s = 0; s < kNumStages; ++s) {
        const auto& st = r.stages[static_cast<std::size_t>(s)];
        os << std::left << std::setw(26) << kStageNames[static_cast<std::size_t>(s)] << std::right
           << std::setw(14) << st.avgMs() << std::setw(14) << st.totalMs << std::setw(15)
           << 100.0 * r.utilisation(s) << "\n";
    }
    os << "\nCompute-stage balance (slowest/fastest): " << r.balanceRatio() << "\n";
}

}  // namespace ipp
