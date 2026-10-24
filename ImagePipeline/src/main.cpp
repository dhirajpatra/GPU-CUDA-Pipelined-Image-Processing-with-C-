// main.cpp - command-line front end.
#include <cstdlib>
#include <iostream>
#include <string>

#include "Pipeline.h"

namespace {

void usage() {
    std::cout << "Usage: ImagePipeline <input_dir> <output_dir> [options]\n"
                 "  --threshold N   edge threshold on |Laplacian|, 0-255   (default 20)\n"
                 "  --sigma S       Gaussian sigma, > 0                    (default 1.0)\n"
                 "  --min-area N    smallest object, in edge pixels        (default 16)\n"
                 "  --slots N       frames in flight; 1 = serial baseline  (default 3)\n"
                 "  --device N      CUDA device index                      (default 0)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }
    try {
        ipp::Pipeline::Config cfg;
        cfg.inputDir = argv[1];
        cfg.outputDir = argv[2];
        for (int i = 3; i < argc; ++i) {
            const std::string opt = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + opt);
            const std::string val = argv[++i];
            if (opt == "--threshold") {
                const int v = std::stoi(val);
                if (v < 0 || v > 255) throw std::invalid_argument("threshold must be 0-255");
                cfg.threshold = static_cast<std::uint8_t>(v);
            } else if (opt == "--sigma") cfg.sigma = std::stof(val);
            else if (opt == "--min-area") cfg.minArea = static_cast<std::uint32_t>(std::stoul(val));
            else if (opt == "--slots") cfg.slots = std::stoi(val);
            else if (opt == "--device") cfg.device = std::stoi(val);
            else throw std::invalid_argument("unknown option " + opt);
        }
        ipp::Pipeline pipeline(cfg);
        const ipp::PipelineReport report = pipeline.run();
        ipp::printReport(report, std::cout);
        std::cout << "\nOutputs written to " << cfg.outputDir.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage();
        return 1;
    }
}
