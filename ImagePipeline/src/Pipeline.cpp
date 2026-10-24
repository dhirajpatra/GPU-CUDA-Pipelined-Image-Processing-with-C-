#include "Pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "BlockingQueue.h"
#include "Bmp.h"
#include "CudaUtils.h"
#include "Frame.h"
#include "Kernels.h"

namespace ipp {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct InputFile {
    fs::path path;
    int width;
    int height;
};

double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::string lowerCase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// Lists *.bmp files (sorted by name) and reads their headers. Unreadable files are
/// skipped with a warning so one bad file cannot stop the whole batch.
std::vector<InputFile> scanInputs(const fs::path& dir, std::size_t& skipped) {
    if (!fs::is_directory(dir)) throw std::runtime_error("input directory not found: " + dir.string());
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && lowerCase(entry.path().extension().string()) == ".bmp") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    std::vector<InputFile> files;
    for (const auto& p : paths) {
        try {
            InputFile f{p, 0, 0};
            readBmpDimensions(p, f.width, f.height);
            files.push_back(std::move(f));
        } catch (const std::exception& e) {
            std::cerr << "warning: skipping " << e.what() << "\n";
            ++skipped;
        }
    }
    return files;
}

}  // namespace

Pipeline::Pipeline(Config config) : config_(std::move(config)) {
    if (config_.slots < 1) throw std::invalid_argument("slots must be >= 1");
    if (!(config_.sigma > 0.0f)) throw std::invalid_argument("sigma must be > 0");
}

PipelineReport Pipeline::run() {
    PipelineReport report;
    for (int s = 0; s < kNumStages; ++s) report.stages[static_cast<std::size_t>(s)] = StageStats{};

    const std::vector<InputFile> files = scanInputs(config_.inputDir, report.skippedFiles);
    if (files.empty()) throw std::runtime_error("no readable .bmp files in " + config_.inputDir.string());
    fs::create_directories(config_.outputDir);

    int deviceCount = 0;
    IPP_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (config_.device < 0 || config_.device >= deviceCount) throw std::runtime_error("invalid CUDA device index");
    IPP_CUDA_CHECK(cudaSetDevice(config_.device));
    setGaussianSigma(config_.sigma);

    // Every buffer is sized once for the largest image, so there is no cudaMalloc/cudaFree
    // (both implicitly synchronise the device) while the pipeline is running.
    std::size_t maxPixels = 0;
    for (const auto& f : files) {
        maxPixels = std::max(maxPixels, static_cast<std::size_t>(f.width) * static_cast<std::size_t>(f.height));
    }
    std::vector<std::unique_ptr<Frame>> frames;
    BlockingQueue<Frame*> freeQueue;
    for (int i = 0; i < config_.slots; ++i) {
        frames.push_back(std::make_unique<Frame>());
        frames.back()->allocate(maxPixels);
        freeQueue.push(frames.back().get());
    }
    ComponentAnalyzer analyzer(maxPixels);

    CudaStream copyStream, gaussStream, edgeStream, finalStream;
    BlockingQueue<Frame*> loadedQ, blurredQ, edgedQ, doneQ;

    std::ofstream summary(config_.outputDir / "summary.csv");
    std::ofstream centroids(config_.outputDir / "centroids.csv");
    if (!summary || !centroids) throw std::runtime_error("cannot create CSV files in output directory");
    summary << "image,width,height,object_count\n";
    centroids << "image,object_id,centroid_x,centroid_y,edge_pixels\n";
    summary.precision(2);
    centroids.precision(2);
    summary << std::fixed;
    centroids << std::fixed;

    // ---- error handling: the first failure is stored, all queues are aborted so every thread unblocks.
    std::mutex errorMutex;
    std::exception_ptr firstError;
    auto abortAll = [&] {
        freeQueue.abort(); loadedQ.abort(); blurredQ.abort(); edgedQ.abort(); doneQ.abort();
    };
    std::vector<std::thread> threads;
    auto spawn = [&](auto body) {
        threads.emplace_back([&, body]() mutable {
            try {
                IPP_CUDA_CHECK(cudaSetDevice(config_.device));
                body();
            } catch (...) {
                { std::lock_guard<std::mutex> lock(errorMutex); if (!firstError) firstError = std::current_exception(); }
                abortAll();
            }
        });
    };

    // Generic compute stage: wait (on the GPU, not the CPU) for the previous stage's event,
    // enqueue this stage's work, record timing events, hand the frame on.
    auto computeStage = [&](BlockingQueue<Frame*>& in, BlockingQueue<Frame*>& out, cudaStream_t stream, int stage,
                            auto work) {
        while (auto item = in.pop()) {
            Frame* f = *item;
            IPP_CUDA_CHECK(cudaStreamWaitEvent(stream, f->events[stage - 1].stop.get(), 0));
            f->events[stage].start.record(stream);
            work(*f);
            f->events[stage].stop.record(stream);
            if (!out.push(f)) break;
        }
        out.close();
    };

    const Clock::time_point wallStart = Clock::now();

    // ---- Stage 0: load BMP on the CPU, enqueue async H2D copy.
    spawn([&] {
        for (std::size_t i = 0; i < files.size(); ++i) {
            auto slot = freeQueue.pop();
            if (!slot) return;
            Frame* f = *slot;
            f->index = i;
            f->name = files[i].path.stem().string();
            f->startedAt = Clock::now();
            readBmpInto(files[i].path, f->hostIn.get(), f->hostIn.capacity(), f->width, f->height);
            f->readMs = msSince(f->startedAt);
            f->events[kStageH2D].start.record(copyStream.get());
            IPP_CUDA_CHECK(cudaMemcpyAsync(f->dBgr.get(), f->hostIn.get(), f->pixels() * 3, cudaMemcpyHostToDevice,
                                           copyStream.get()));
            f->events[kStageH2D].stop.record(copyStream.get());
            if (!loadedQ.push(f)) return;
        }
        loadedQ.close();
    });

    // ---- Stage 1: Gaussian.
    spawn([&] {
        computeStage(loadedQ, blurredQ, gaussStream.get(), kStageGaussian, [&](Frame& f) {
            launchGaussian(f.dBgr.get(), f.dGray.get(), f.width, f.height, gaussStream.get());
        });
    });

    // ---- Stage 2: Laplacian.
    spawn([&] {
        computeStage(blurredQ, edgedQ, edgeStream.get(), kStageEdge, [&](Frame& f) {
            launchLaplacian(f.dGray.get(), f.dEdge.get(), f.width, f.height, edgeStream.get());
        });
    });

    // ---- Stage 3: threshold + labeling + overlay, then D2H copy of the result.
    spawn([&] {
        const cudaStream_t s = finalStream.get();
        while (auto item = edgedQ.pop()) {
            Frame* f = *item;
            IPP_CUDA_CHECK(cudaStreamWaitEvent(s, f->events[kStageEdge].stop.get(), 0));
            f->events[kStageFinal].start.record(s);
            f->result = analyzer.analyze(f->dEdge.get(), f->dBgr.get(), f->width, f->height, config_.threshold,
                                         config_.minArea, s);
            f->events[kStageFinal].stop.record(s);
            f->events[kStageD2H].start.record(s);
            IPP_CUDA_CHECK(cudaMemcpyAsync(f->hostOut.get(), f->dBgr.get(), f->pixels() * 3, cudaMemcpyDeviceToHost, s));
            f->events[kStageD2H].stop.record(s);
            IPP_CUDA_CHECK(cudaStreamSynchronize(s));
            if (!doneQ.push(f)) break;
        }
        doneQ.close();
    });

    // ---- Writer: encode output image, append CSV rows, collect timings, recycle the slot.
    spawn([&] {
        while (auto item = doneQ.pop()) {
            Frame* f = *item;
            const Clock::time_point writeStart = Clock::now();
            writeBmp(config_.outputDir / (f->name + "_edges.bmp"), f->hostOut.get(), f->width, f->height);

            summary << f->name << ',' << f->width << ',' << f->height << ',' << f->result.totalObjects << '\n';
            for (std::size_t k = 0; k < f->result.objects.size(); ++k) {
                const Detection& d = f->result.objects[k];
                centroids << f->name << ',' << k + 1 << ',' << d.cx << ',' << d.cy << ',' << d.area << '\n';
            }
            if (f->result.truncated) {
                std::cerr << "warning: " << f->name << ": " << f->result.totalObjects
                          << " objects found, only the first " << f->result.objects.size() << " listed\n";
            }

            for (int s = 0; s < kNumStages; ++s) {
                StageStats& st = report.stages[static_cast<std::size_t>(s)];
                st.totalMs += CudaEvent::elapsedMs(f->events[s].start, f->events[s].stop);
                ++st.frames;
            }
            report.totalReadMs += f->readMs;
            report.totalWriteMs += msSince(writeStart);
            report.totalLatencyMs += msSince(f->startedAt);
            ++report.frames;
            if (!freeQueue.push(f)) break;
        }
    });

    for (auto& t : threads) t.join();
    report.wallMs = msSince(wallStart);
    if (firstError) std::rethrow_exception(firstError);

    // Metrics file for the design document.
    std::ofstream metrics(config_.outputDir / "metrics.csv");
    metrics << "stage,frames,total_ms,avg_ms,gpu_busy_pct_of_wall\n";
    for (int s = 0; s < kNumStages; ++s) {
        const StageStats& st = report.stages[static_cast<std::size_t>(s)];
        metrics << kStageNames[static_cast<std::size_t>(s)] << ',' << st.frames << ',' << st.totalMs << ',' << st.avgMs()
                << ',' << 100.0 * report.utilisation(s) << '\n';
    }
    return report;
}

}  // namespace ipp
