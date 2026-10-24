// Tests.cpp - self-contained unit/integration tests (no external framework).
// GPU tests are skipped (not failed) when no CUDA device is present.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "BlockingQueue.h"
#include "Bmp.h"
#include "Kernels.h"
#include "Pipeline.h"

namespace fs = std::filesystem;
using namespace ipp;

// ------------------------------------------------------------------ mini framework
struct Skip {};
struct Failure { std::string message; };
struct TestCase { const char* name; std::function<void()> body; };
static std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
struct Registrar { Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };

#define TEST(name) \
    static void name(); \
    static Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond) do { if (!(cond)) throw Failure{std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": CHECK(" #cond ")"}; } while (0)
#define CHECK_EQ(a, b) do { const auto va_ = (a); const auto vb_ = (b); if (!(va_ == vb_)) { std::ostringstream o_; \
    o_ << __FILE__ << ":" << __LINE__ << ": " #a " == " #b " (" << va_ << " vs " << vb_ << ")"; throw Failure{o_.str()}; } } while (0)
#define CHECK_NEAR(a, b, tol) do { const double va_ = (a), vb_ = (b); if (std::fabs(va_ - vb_) > (tol)) { std::ostringstream o_; \
    o_ << __FILE__ << ":" << __LINE__ << ": " #a " ~ " #b " (" << va_ << " vs " << vb_ << ")"; throw Failure{o_.str()}; } } while (0)

static void requireGpu() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) throw Skip{};
}

// ------------------------------------------------------------------ helpers
template <typename T>
static DeviceBuffer<T> toDevice(const std::vector<T>& v) {
    DeviceBuffer<T> d;
    d.allocate(v.empty() ? 1 : v.size());
    IPP_CUDA_CHECK(cudaMemcpy(d.get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> toHost(const DeviceBuffer<T>& d, std::size_t n) {
    std::vector<T> v(n);
    IPP_CUDA_CHECK(cudaMemcpy(v.data(), d.get(), n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static std::vector<std::uint8_t> grayToBgr(const std::vector<std::uint8_t>& g) {
    std::vector<std::uint8_t> bgr(g.size() * 3);
    for (std::size_t i = 0; i < g.size(); ++i) bgr[3 * i] = bgr[3 * i + 1] = bgr[3 * i + 2] = g[i];
    return bgr;
}
static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

static std::vector<std::uint8_t> runGaussian(const std::vector<std::uint8_t>& gray, int w, int h, float sigma) {
    setGaussianSigma(sigma);
    auto dIn = toDevice(grayToBgr(gray));
    DeviceBuffer<std::uint8_t> dOut;
    dOut.allocate(gray.size());
    launchGaussian(dIn.get(), dOut.get(), w, h, nullptr);
    IPP_CUDA_CHECK(cudaDeviceSynchronize());
    return toHost(dOut, gray.size());
}
static std::vector<std::uint8_t> runLaplacian(const std::vector<std::uint8_t>& gray, int w, int h) {
    auto dIn = toDevice(gray);
    DeviceBuffer<std::uint8_t> dOut;
    dOut.allocate(gray.size());
    launchLaplacian(dIn.get(), dOut.get(), w, h, nullptr);
    IPP_CUDA_CHECK(cudaDeviceSynchronize());
    return toHost(dOut, gray.size());
}

struct AnalysisRun {
    AnalysisResult result;
    std::vector<std::uint8_t> overlay;  // BGR after analysis
};
static AnalysisRun runAnalysis(const std::vector<std::uint8_t>& edge, int w, int h, std::uint32_t minArea) {
    ComponentAnalyzer analyzer(edge.size());
    auto dEdge = toDevice(edge);
    auto dBgr = toDevice(std::vector<std::uint8_t>(edge.size() * 3, 40));
    AnalysisRun run;
    run.result = analyzer.analyze(dEdge.get(), dBgr.get(), w, h, 128, minArea, nullptr);
    IPP_CUDA_CHECK(cudaDeviceSynchronize());
    run.overlay = toHost(dBgr, edge.size() * 3);
    return run;
}
static void drawSquareOutline(std::vector<std::uint8_t>& img, int w, int x0, int y0, int x1, int y1) {
    for (int x = x0; x <= x1; ++x) { img[y0 * w + x] = 255; img[y1 * w + x] = 255; }
    for (int y = y0; y <= y1; ++y) { img[y * w + x0] = 255; img[y * w + x1] = 255; }
}

// ------------------------------------------------------------------ BMP
TEST(Bmp_RoundTrip_VariousWidths_CoversRowPadding) {
    const fs::path path = fs::temp_directory_path() / "ipp_rt.bmp";
    for (int w : {1, 2, 3, 4, 5, 33}) {
        Image a; a.width = w; a.height = 7; a.bgr.resize(static_cast<std::size_t>(w) * 7 * 3);
        for (std::size_t i = 0; i < a.bgr.size(); ++i) a.bgr[i] = static_cast<std::uint8_t>(i * 37 + 11);
        writeBmp(path, a.bgr.data(), a.width, a.height);
        const Image b = readBmp(path);
        CHECK_EQ(b.width, a.width); CHECK_EQ(b.height, a.height); CHECK(b.bgr == a.bgr);
    }
    fs::remove(path);
}
TEST(Bmp_Reads8BitPalettised) {
    const fs::path path = fs::temp_directory_path() / "ipp_p8.bmp";
    // 3x2, bottom-up, palette: 0=black 1=red(BGR 0,0,255) 2=blue(BGR 255,0,0)
    std::vector<std::uint8_t> f(14 + 40 + 3 * 4 + 8, 0);
    auto put32 = [&](std::size_t o, std::uint32_t v) { for (int i = 0; i < 4; ++i) f[o + i] = static_cast<std::uint8_t>(v >> (8 * i)); };
    f[0] = 'B'; f[1] = 'M'; put32(2, static_cast<std::uint32_t>(f.size())); put32(10, 14 + 40 + 12);
    put32(14, 40); put32(18, 3); put32(22, 2); f[26] = 1; f[28] = 8; put32(46, 3);
    const std::size_t pal = 14 + 40;
    f[pal + 4 + 2] = 255;  // entry1: R
    f[pal + 8 + 0] = 255;  // entry2: B
    const std::size_t px = pal + 12;
    f[px + 0] = 1; f[px + 1] = 1; f[px + 2] = 1;  // bottom row: red red red
    f[px + 4] = 0; f[px + 5] = 2; f[px + 6] = 0;  // top row: black blue black
    { std::ofstream o(path, std::ios::binary); o.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size())); }
    const Image img = readBmp(path);
    CHECK_EQ(img.width, 3); CHECK_EQ(img.height, 2);
    CHECK_EQ(static_cast<int>(img.bgr[3 * 1 + 0]), 255);        // top row, x=1 is blue
    CHECK_EQ(static_cast<int>(img.bgr[(3 + 0) * 3 + 2]), 255);  // bottom row, x=0 is red
    fs::remove(path);
}
TEST(Bmp_RejectsGarbageAndTruncation) {
    const fs::path path = fs::temp_directory_path() / "ipp_bad.bmp";
    { std::ofstream o(path, std::ios::binary); o << "not a bitmap at all, definitely longer than 54 bytes of header data....."; }
    bool threw = false; try { readBmp(path); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    std::vector<std::uint8_t> px(3 * 4 * 4, 9);
    writeBmp(path, px.data(), 4, 4);
    fs::resize_file(path, 54 + 20);  // cut pixel data short
    threw = false; try { readBmp(path); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    fs::remove(path);
}

// ------------------------------------------------------------------ queue
TEST(BlockingQueue_DrainsAfterClose_AndRejectsPush) {
    BlockingQueue<int> q(2);
    long sum = 0;
    std::thread consumer([&] { while (auto v = q.pop()) sum += *v; });
    for (int i = 1; i <= 1000; ++i) CHECK(q.push(i));
    q.close();
    consumer.join();
    CHECK_EQ(sum, 500500L);
    CHECK(!q.push(1));
}

// ------------------------------------------------------------------ Gaussian
TEST(Gaussian_Kernel_IsNormalisedAndSymmetric) {
    const auto k = makeGaussianKernel3x3(1.0f);
    float sum = 0; for (float v : k) sum += v;
    CHECK_NEAR(sum, 1.0, 1e-6);
    CHECK_NEAR(k[0], k[2], 1e-7); CHECK_NEAR(k[1], k[7], 1e-7); CHECK_NEAR(k[3], k[5], 1e-7);
    CHECK(k[4] > k[1] && k[1] > k[0]);
}
TEST(Gaussian_ConstantImage_IsPreserved_OddSize) {
    requireGpu();
    const int w = 37, h = 23;
    const auto out = runGaussian(std::vector<std::uint8_t>(w * h, 123), w, h, 1.0f);
    for (auto v : out) CHECK_EQ(static_cast<int>(v), 123);
}
TEST(Gaussian_MatchesCpuReference_RandomImage) {
    requireGpu();
    const int w = 61, h = 45;  // not multiples of the block size
    std::mt19937 rng(1);
    std::vector<std::uint8_t> g(w * h);
    for (auto& v : g) v = static_cast<std::uint8_t>(rng());
    const auto k = makeGaussianKernel3x3(1.0f);
    const auto gpu = runGaussian(g, w, h, 1.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    acc += k[static_cast<std::size_t>((dy + 1) * 3 + dx + 1)] * g[clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)];
            CHECK(std::abs(static_cast<int>(gpu[y * w + x]) - static_cast<int>(acc + 0.5f)) <= 1);
        }
}
TEST(Gaussian_OneByOneImage) {
    requireGpu();
    const auto out = runGaussian({77}, 1, 1, 1.0f);
    CHECK_EQ(static_cast<int>(out[0]), 77);
}

// ------------------------------------------------------------------ Laplacian
TEST(Laplacian_ConstantImage_IsZero) {
    requireGpu();
    for (auto v : runLaplacian(std::vector<std::uint8_t>(40 * 17, 90), 40, 17)) CHECK_EQ(static_cast<int>(v), 0);
}
TEST(Laplacian_VerticalStep_KnownValues) {
    requireGpu();
    const int w = 8, h = 3;
    std::vector<std::uint8_t> g(w * h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) g[y * w + x] = x < 4 ? 0 : 100;
    const auto out = runLaplacian(g, w, h);
    for (int y = 0; y < h; ++y) {
        CHECK_EQ(static_cast<int>(out[y * w + 3]), 100);  // |0+100-0| on the dark side of the step
        CHECK_EQ(static_cast<int>(out[y * w + 4]), 100);
        CHECK_EQ(static_cast<int>(out[y * w + 1]), 0);
        CHECK_EQ(static_cast<int>(out[y * w + 6]), 0);
    }
}
TEST(Laplacian_MatchesCpuReference_Exactly) {
    requireGpu();
    const int w = 70, h = 33;
    std::mt19937 rng(2);
    std::vector<std::uint8_t> g(w * h);
    for (auto& v : g) v = static_cast<std::uint8_t>(rng());
    const auto gpu = runLaplacian(g, w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto at = [&](int xx, int yy) { return static_cast<int>(g[clampi(yy, 0, h - 1) * w + clampi(xx, 0, w - 1)]); };
            const int r = std::min(255, std::abs(at(x, y - 1) + at(x, y + 1) + at(x - 1, y) + at(x + 1, y) - 4 * at(x, y)));
            CHECK_EQ(static_cast<int>(gpu[y * w + x]), r);
        }
}

// ------------------------------------------------------------------ components
TEST(Components_TwoSquares_CountsCentroidsAndOverlay) {
    requireGpu();
    const int w = 50, h = 20;
    std::vector<std::uint8_t> e(w * h, 0);
    drawSquareOutline(e, w, 5, 5, 14, 14);
    drawSquareOutline(e, w, 30, 5, 39, 14);
    const auto run = runAnalysis(e, w, h, 1);
    CHECK_EQ(run.result.totalObjects, 2u);
    CHECK_EQ(run.result.objects.size(), std::size_t{2});
    CHECK_NEAR(run.result.objects[0].cx, 9.5, 0.01); CHECK_NEAR(run.result.objects[0].cy, 9.5, 0.01);
    CHECK_NEAR(run.result.objects[1].cx, 34.5, 0.01);
    CHECK_EQ(run.result.objects[0].area, 36u);
    // An edge pixel (not on the centroid cross) is recoloured; a far background pixel is untouched.
    const int edgeIdx = 5 * w + 7;
    CHECK(run.overlay[edgeIdx * 3] != 40 || run.overlay[edgeIdx * 3 + 1] != 40 || run.overlay[edgeIdx * 3 + 2] != 40);
    CHECK_EQ(static_cast<int>(run.overlay[(19 * w + 49) * 3]), 40);
    // A white centroid cross is drawn inside the first square (9.5 rounds to pixel 9 or 10).
    int whites = 0;
    for (int y = 7; y <= 12; ++y)
        for (int x = 7; x <= 12; ++x)
            if (run.overlay[(y * w + x) * 3] == 255 && run.overlay[(y * w + x) * 3 + 1] == 255 &&
                run.overlay[(y * w + x) * 3 + 2] == 255) ++whites;
    CHECK(whites >= 6);
}
TEST(Components_MinAreaFiltersNoise) {
    requireGpu();
    const int w = 30, h = 30;
    std::vector<std::uint8_t> e(w * h, 0);
    e[3 * w + 3] = 255; e[10 * w + 20] = 255;     // single-pixel noise
    drawSquareOutline(e, w, 12, 12, 20, 20);      // real object (32 px)
    const auto run = runAnalysis(e, w, h, 8);
    CHECK_EQ(run.result.totalObjects, 1u);
    CHECK_EQ(static_cast<int>(run.overlay[(3 * w + 3) * 3]), 40);  // noise not painted
}
TEST(Components_DiagonalPixels_AreOneObject_EightConnectivity) {
    requireGpu();
    const int w = 16, h = 16;
    std::vector<std::uint8_t> e(w * h, 0);
    for (int i = 2; i < 12; ++i) e[i * w + i] = 255;
    const auto run = runAnalysis(e, w, h, 1);
    CHECK_EQ(run.result.totalObjects, 1u);
    CHECK_EQ(run.result.objects[0].area, 10u);
}
TEST(Components_LongSpiral_ConvergesToSingleObject) {
    requireGpu();
    const int w = 96, h = 96;
    std::vector<std::uint8_t> e(w * h, 0);
    auto mark = [&](int x, int y) { return x >= 0 && y >= 0 && x < w && y < h && e[y * w + x] != 0; };
    int x = 0, y = 0, dx = 1, dy = 0, turns = 0;
    while (turns < 2) {
        e[y * w + x] = 255;
        const int nx = x + dx, ny = y + dy;
        const bool inBounds = nx >= 0 && ny >= 0 && nx < w && ny < h;
        if (inBounds && !mark(nx, ny) && !mark(nx + dx, ny + dy)) { x = nx; y = ny; turns = 0; }
        else { const int t = dx; dx = -dy; dy = t; ++turns; }  // turn right
    }
    std::uint32_t pixels = 0; for (auto v : e) pixels += v != 0;
    CHECK(pixels > 1000);
    const auto run = runAnalysis(e, w, h, 1);
    CHECK_EQ(run.result.totalObjects, 1u);
    CHECK_EQ(run.result.objects[0].area, pixels);
}
TEST(Components_EmptyImage_NoObjects) {
    requireGpu();
    const auto run = runAnalysis(std::vector<std::uint8_t>(20 * 20, 0), 20, 20, 1);
    CHECK_EQ(run.result.totalObjects, 0u);
    CHECK(run.result.objects.empty());
}
TEST(Components_ObjectTouchingImageBorder) {
    requireGpu();
    const int w = 12, h = 12;
    std::vector<std::uint8_t> e(w * h, 0);
    drawSquareOutline(e, w, 0, 0, 5, 5);
    const auto run = runAnalysis(e, w, h, 1);
    CHECK_EQ(run.result.totalObjects, 1u);
    CHECK_NEAR(run.result.objects[0].cx, 2.5, 0.01);
}

// ------------------------------------------------------------------ end to end
static void fillRect(Image& img, int x0, int y0, int x1, int y1, std::uint8_t v) {
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x)
        for (int c = 0; c < 3; ++c) img.bgr[(static_cast<std::size_t>(y) * img.width + x) * 3 + c] = v;
}
static Image blank(int w, int h) { Image i; i.width = w; i.height = h; i.bgr.assign(static_cast<std::size_t>(w) * h * 3, 20); return i; }
static std::string slurp(const fs::path& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

TEST(Pipeline_EndToEnd_MixedResolutions_SerialEqualsPipelined) {
    requireGpu();
    const fs::path root = fs::temp_directory_path() / "ipp_e2e";
    fs::remove_all(root); fs::create_directories(root / "in");
    { Image a = blank(64, 48);   fillRect(a, 8, 8, 24, 24, 220);  fillRect(a, 38, 10, 54, 36, 220);   writeBmp(root / "in/a.bmp", a.bgr.data(), a.width, a.height); }
    { Image b = blank(200, 120); fillRect(b, 10, 10, 50, 50, 220); fillRect(b, 80, 20, 120, 60, 220); fillRect(b, 140, 70, 190, 110, 220); writeBmp(root / "in/b.bmp", b.bgr.data(), b.width, b.height); }
    { Image c = blank(33, 17);   fillRect(c, 8, 4, 24, 13, 220);  writeBmp(root / "in/c.bmp", c.bgr.data(), c.width, c.height); }
    { std::ofstream bad(root / "in/zz_corrupt.bmp"); bad << "garbage"; }  // must be skipped, not fatal

    Pipeline::Config cfg;
    cfg.inputDir = root / "in"; cfg.threshold = 20; cfg.minArea = 8; cfg.sigma = 1.0f;
    cfg.outputDir = root / "out3"; cfg.slots = 3;
    const PipelineReport r3 = Pipeline(cfg).run();
    cfg.outputDir = root / "out1"; cfg.slots = 1;
    const PipelineReport r1 = Pipeline(cfg).run();

    CHECK_EQ(r3.frames, std::size_t{3}); CHECK_EQ(r3.skippedFiles, std::size_t{1}); CHECK_EQ(r1.frames, std::size_t{3});
    CHECK_EQ(slurp(root / "out3/summary.csv"), std::string("image,width,height,object_count\na,64,48,2\nb,200,120,3\nc,33,17,1\n"));
    CHECK_EQ(slurp(root / "out3/summary.csv"), slurp(root / "out1/summary.csv"));
    CHECK_EQ(slurp(root / "out3/centroids.csv"), slurp(root / "out1/centroids.csv"));
    for (const char* n : {"a", "b", "c"}) {
        const Image out = readBmp(root / "out3" / (std::string(n) + "_edges.bmp"));
        const Image in = readBmp(root / "in" / (std::string(n) + ".bmp"));
        CHECK_EQ(out.width, in.width); CHECK_EQ(out.height, in.height);
        CHECK(out.bgr != in.bgr);  // overlay actually changed something
    }
    CHECK(r3.stages[kStageGaussian].frames == 3 && r3.stages[kStageGaussian].totalMs > 0.0);
    fs::remove_all(root);
}
TEST(Pipeline_EmptyDirectory_Throws) {
    requireGpu();
    const fs::path root = fs::temp_directory_path() / "ipp_empty";
    fs::remove_all(root); fs::create_directories(root / "in");
    Pipeline::Config cfg; cfg.inputDir = root / "in"; cfg.outputDir = root / "out";
    bool threw = false; try { Pipeline(cfg).run(); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    fs::remove_all(root);
}

// ------------------------------------------------------------------ runner
int main() {
    int passed = 0, failed = 0, skipped = 0;
    for (const auto& t : registry()) {
        try { t.body(); ++passed; std::cout << "[ PASS ] " << t.name << "\n"; }
        catch (const Skip&) { ++skipped; std::cout << "[ SKIP ] " << t.name << " (no CUDA device)\n"; }
        catch (const Failure& f) { ++failed; std::cout << "[ FAIL ] " << t.name << "\n         " << f.message << "\n"; }
        catch (const std::exception& e) { ++failed; std::cout << "[ FAIL ] " << t.name << "\n         exception: " << e.what() << "\n"; }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed, " << skipped << " skipped\n";
    return failed == 0 ? 0 : 1;
}
