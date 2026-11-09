// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// picBunchBench: synthetic scaling instrument for the picBunch field pipeline.
//
// The benchmark builds the bunch in memory (no openPMD input), calls the public
// picBunch::computeField() entry point with a StageTimes* and reports the
// per-stage wall time, the grid that makeGrid() selected, and peak memory.
// It does not change the compute core or any public API.

#include "picBunch/Compute.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/Types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef PICBUNCH_BENCH_HAVE_CUDA
#    include <cuda_runtime.h>
#endif

namespace
{
    // ---------------------------------------------------------------- memory

    std::size_t hostPeakMiB()
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while(std::getline(status, line))
            if(line.rfind("VmHWM:", 0) == 0)
            {
                double kb = 0.0;
                std::sscanf(line.c_str() + 6, "%lf", &kb);
                return static_cast<std::size_t>(kb / 1024.0);
            }
        return 0;
    }

    void resetHostPeak()
    {
        // Linux >= 4.0: writing 5 to clear_refs resets the peak RSS counter.
        std::ofstream clear("/proc/self/clear_refs");
        if(clear)
            clear << "5\n";
    }

#ifdef PICBUNCH_BENCH_HAVE_CUDA
    //! cudaMemGetInfo with the error checked, so a failed query never silently
    //! becomes a bogus baseline or peak.
    void cudaMemGetInfoChecked(std::size_t& freeBytes, std::size_t& totalBytes, char const* context)
    {
        cudaError_t const status = cudaMemGetInfo(&freeBytes, &totalBytes);
        if(status != cudaSuccess)
            throw std::runtime_error(
                std::string("picBunchBench: cudaMemGetInfo (") + context + "): "
                + cudaGetErrorString(status));
    }

    struct GpuMemorySampler
    {
        std::atomic<bool> stop{false};
        std::atomic<std::size_t> minFree{std::numeric_limits<std::size_t>::max()};
        std::thread worker;

        void start()
        {
            worker = std::thread([this] {
                while(!stop.load(std::memory_order_relaxed))
                {
                    std::size_t freeBytes = 0;
                    std::size_t totalBytes = 0;
                    if(cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess)
                    {
                        std::size_t current = minFree.load(std::memory_order_relaxed);
                        while(freeBytes < current
                              && !minFree.compare_exchange_weak(current, freeBytes))
                        {
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            });
        }

        //! Peak device bytes above the level observed when start() was called.
        std::size_t stopAndPeakBytes(std::size_t baselineFree) const
        {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            cudaMemGetInfoChecked(freeBytes, totalBytes, "peak sample");
            std::size_t const peakFree = minFree.load();
            if(peakFree == std::numeric_limits<std::size_t>::max())
                return 0;
            return baselineFree > peakFree ? baselineFree - peakFree : 0;
        }
    };
#endif

    // ---------------------------------------------------------------- bunch

    //! R3 low-discrepancy point in [0,1)^3 (plastic-number sequence).
    std::array<double, 3> r3(std::size_t index)
    {
        double const phi = 1.32471795724474602596;
        double const a1 = 1.0 / phi;
        double const a2 = 1.0 / (phi * phi);
        double const a3 = 1.0 / (phi * phi * phi);
        auto frac = [](double v) { return v - std::floor(v); };
        double const i = static_cast<double>(index + 1u);
        return {frac(0.5 + a1 * i), frac(0.5 + a2 * i), frac(0.5 + a3 * i)};
    }

    //! Uniform fill of [0, box]^3 with one shared momentum along +z.
    picBunch::Bunch makeBunch(std::size_t count, double box, double gamma)
    {
        picBunch::Bunch bunch;
        bunch.resize(count);
        double const uz = std::sqrt(std::max(0.0, gamma * gamma - 1.0));
        for(std::size_t p = 0; p < count; ++p)
        {
            auto const r = r3(p);
            bunch.x[p] = r[0] * box;
            bunch.y[p] = r[1] * box;
            bunch.z[p] = r[2] * box;
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = uz;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = picBunch::kElementaryCharge;
        }
        return bunch;
    }

    //! A large empty grid: two zero-charge particles pin the bounding box.
    picBunch::Bunch makeProbeBunch(double box, double gamma, double px, double py, double pz)
    {
        picBunch::Bunch bunch;
        bunch.resize(3);
        double const uz = std::sqrt(std::max(0.0, gamma * gamma - 1.0));
        for(std::size_t p = 0; p < 3; ++p)
        {
            bunch.ux[p] = 0.0;
            bunch.uy[p] = 0.0;
            bunch.uz[p] = uz;
        }
        // zero-charge corners: no deposited source, only the bounding box
        bunch.x[0] = 0.0;
        bunch.y[0] = 0.0;
        bunch.z[0] = 0.0;
        bunch.weight[0] = 0.0;
        bunch.charge[0] = 0.0;
        bunch.x[1] = box;
        bunch.y[1] = box;
        bunch.z[1] = box;
        bunch.weight[1] = 0.0;
        bunch.charge[1] = 0.0;
        // the physical source
        bunch.x[2] = px;
        bunch.y[2] = py;
        bunch.z[2] = pz;
        bunch.weight[2] = 1.0;
        bunch.charge[2] = picBunch::kElementaryCharge;
        return bunch;
    }

    // ---------------------------------------------------------------- stats

    double median(std::vector<double> values)
    {
        if(values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        std::size_t const n = values.size();
        return (n % 2u) ? values[n / 2u] : 0.5 * (values[n / 2u - 1u] + values[n / 2u]);
    }

    struct Row
    {
        std::size_t particles = 0;
        double box = 0.0;
        double gamma = 0.0;
        double spacing = 0.0;
        unsigned padding = 2;
        std::array<std::size_t, 3> n{0, 0, 0};
        std::array<std::size_t, 3> m{0, 0, 0};
        double planMin = 0.0;
        double planMedian = 0.0;
        double depositMin = 0.0;
        double depositMedian = 0.0;
        double kernelMin = 0.0;
        double kernelMedian = 0.0;
        double fftMin = 0.0;
        double fftMedian = 0.0;
        double readoutMin = 0.0;
        double readoutMedian = 0.0;
        double totalMin = 0.0;
        double totalMedian = 0.0;
        double totalMax = 0.0;
        double deviceAllocMiB = 0.0;
        std::size_t peakDeviceMiB = 0;
        std::size_t peakHostMiB = 0;
    };

    void writeRawHeader(std::FILE* out)
    {
        std::fprintf(
            out,
            "row_id,repeat,phase,clock_mhz,plan_ms,deposit_ms,kernel_ms,fft_ms,"
            "readout_ms,total_ms\n");
    }

    void writeRawRow(
        std::FILE* out,
        std::string const& rowId,
        int repeat,
        char const* phase,
        picBunch::StageTimes const& t)
    {
        double const total = t.depositMs + t.kernelMs + t.fftMs + t.readoutMs;
        std::fprintf(
            out,
            "%s,%d,%s,,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            rowId.c_str(),
            repeat,
            phase,
            t.planMs,
            t.depositMs,
            t.kernelMs,
            t.fftMs,
            t.readoutMs,
            total);
    }

    Row runConfiguration(
        std::size_t particles,
        double box,
        double gamma,
        double spacing,
        unsigned padding,
        int repeats,
        int warmups = 2,
        std::FILE* raw = nullptr,
        std::string const& rowId = {})
    {
        picBunch::Bunch const bunch = makeBunch(particles, box, gamma);
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {spacing, spacing, spacing};
        options.fftPadding = padding;
        options.marginCells = 4.0;

        std::vector<picBunch::StageTimes> samples;
        Row row;
        row.particles = particles;
        row.box = box;
        row.gamma = gamma;
        row.spacing = spacing;
        row.padding = padding;
        std::size_t peakDevice = 0;

        // Untimed warm-ups so clocks and first-touch costs do not land in the
        // measured samples.  The documented protocol uses two.
        for(int w = 0; w < warmups; ++w)
        {
            picBunch::StageTimes warm;
            picBunch::Field const warmField = picBunch::computeField(bunch, options, &warm);
            (void) warmField;
            if(raw)
                writeRawRow(raw, rowId, w, "warmup", warm);
        }

        resetHostPeak();
        for(int r = 0; r < repeats; ++r)
        {
            picBunch::StageTimes times;
#ifdef PICBUNCH_BENCH_HAVE_CUDA
            GpuMemorySampler sampler;
            std::size_t freeAtStart = 0;
            std::size_t totalAtStart = 0;
            cudaMemGetInfoChecked(freeAtStart, totalAtStart, "baseline");
            sampler.start();
#endif
            picBunch::Field const field = picBunch::computeField(bunch, options, &times);
#ifdef PICBUNCH_BENCH_HAVE_CUDA
            sampler.stop.store(true);
            sampler.worker.join();
            peakDevice = std::max(peakDevice, sampler.stopAndPeakBytes(freeAtStart));
#endif
            samples.push_back(times);
            if(raw)
                writeRawRow(raw, rowId, r, "timed", times);
            if(r == 0)
            {
                row.n = field.grid.n;
                row.m = field.grid.paddedSize();
            }
        }
        row.peakDeviceMiB = peakDevice / (1024u * 1024u);
        row.peakHostMiB = hostPeakMiB();
        double const mx = static_cast<double>(row.m[0]);
        double const my = static_cast<double>(row.m[1]);
        double const mz = static_cast<double>(row.m[2]);
        double const spec = mz * my * (std::floor(mx / 2.0) + 1.0);
        // exact allocation of computeField: 3 real M^3 arrays, 2 packed
        // complex half-spectra, and 4 per-particle arrays.
        row.deviceAllocMiB
            = (3.0 * mx * my * mz * 8.0 + 2.0 * spec * 16.0
               + 32.0 * static_cast<double>(particles))
            / (1024.0 * 1024.0);

        auto collect = [&](double picBunch::StageTimes::*member) {
            std::vector<double> v;
            v.reserve(samples.size());
            for(auto const& s : samples)
                v.push_back(s.*member);
            return v;
        };
        auto minimum = [](std::vector<double> v) {
            return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
        };

        auto const plan = collect(&picBunch::StageTimes::planMs);
        auto const deposit = collect(&picBunch::StageTimes::depositMs);
        auto const kernel = collect(&picBunch::StageTimes::kernelMs);
        auto const fft = collect(&picBunch::StageTimes::fftMs);
        auto const readout = collect(&picBunch::StageTimes::readoutMs);
        std::vector<double> total;
        for(std::size_t i = 0; i < samples.size(); ++i)
            total.push_back(
                samples[i].depositMs + samples[i].kernelMs + samples[i].fftMs + samples[i].readoutMs);

        row.planMin = minimum(plan);
        row.planMedian = median(plan);
        row.depositMin = minimum(deposit);
        row.depositMedian = median(deposit);
        row.kernelMin = minimum(kernel);
        row.kernelMedian = median(kernel);
        row.fftMin = minimum(fft);
        row.fftMedian = median(fft);
        row.readoutMin = minimum(readout);
        row.readoutMedian = median(readout);
        row.totalMin = minimum(total);
        row.totalMedian = median(total);
        row.totalMax = total.empty() ? 0.0 : *std::max_element(total.begin(), total.end());
        return row;
    }

    void writeCsvHeader(std::FILE* out)
    {
        std::fprintf(
            out,
            "particles,box_cells,gamma,spacing,padding,Mx,My,Mz,nx,ny,nz,"
            "deposit_min_ms,deposit_median_ms,kernel_min_ms,kernel_median_ms,"
            "fft_min_ms,fft_median_ms,readout_min_ms,readout_median_ms,"
            "total_min_ms,total_median_ms,total_max_ms,device_alloc_MiB,peak_device_MiB,"
            "peak_host_MiB,plan_min_ms,plan_median_ms\n");
    }

    void writeCsvRow(std::FILE* out, Row const& r)
    {
        std::fprintf(
            out,
            "%zu,%.17g,%.17g,%.17g,%u,%zu,%zu,%zu,%zu,%zu,%zu,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%zu,%zu,"
            "%.6f,%.6f\n",
            r.particles,
            r.box,
            r.gamma,
            r.spacing,
            r.padding,
            r.m[0],
            r.m[1],
            r.m[2],
            r.n[0],
            r.n[1],
            r.n[2],
            r.depositMin,
            r.depositMedian,
            r.kernelMin,
            r.kernelMedian,
            r.fftMin,
            r.fftMedian,
            r.readoutMin,
            r.readoutMedian,
            r.totalMin,
            r.totalMedian,
            r.totalMax,
            r.deviceAllocMiB,
            r.peakDeviceMiB,
            r.peakHostMiB,
            r.planMin,
            r.planMedian);
    }

    // ------------------------------------------------- Coulomb / TSC reference

    //! Direct TSC-convolved Coulomb field of one particle at a grid node.
    //!
    //! E(node) = q sum_lag W(lag) K_coulomb(node - (particle + lag*h)).
    //! With sufficient zero padding this equals the FFT convolution to roundoff.
    std::array<double, 6> directTscCoulomb(
        picBunch::Grid const& grid,
        double px,
        double py,
        double pz,
        double x,
        double y,
        double z,
        int component,
        int& lagsUsed)
    {
        lagsUsed = 0;
        double const h = grid.spacing[0];
        double const dx = picBunch::kYeeOffset[component][0] * h;
        double const dy = picBunch::kYeeOffset[component][1] * h;
        double const dz = picBunch::kYeeOffset[component][2] * h;
        std::array<double, 6> acc{};
        // The particle is at an integer node, so the TSC stencil is lags -1..1.
        for(int a = -1; a <= 1; ++a)
            for(int b = -1; b <= 1; ++b)
                for(int c = -1; c <= 1; ++c)
                {
                    double const w = picBunch::tscShape(static_cast<double>(a))
                        * picBunch::tscShape(static_cast<double>(b))
                        * picBunch::tscShape(static_cast<double>(c));
                    double const sx = px + static_cast<double>(a) * h;
                    double const sy = py + static_cast<double>(b) * h;
                    double const sz = pz + static_cast<double>(c) * h;
                    std::array<double, 6> kernel{};
                    picBunch::pointFieldNormalized(
                        x + dx - sx,
                        y + dy - sy,
                        z + dz - sz,
                        0.0,
                        0.0,
                        0.0,
                        kernel.data());
                    for(int d = 0; d < 6; ++d)
                        acc[d] += w * kernel[d];
                    ++lagsUsed;
                }
        return acc;
    }

    int runCoulombCheck(std::FILE* csv, bool haveCsv)
    {
        double const box = 64.0;
        double const h = 1.0;
        double const center = box / 2.0;
        picBunch::Bunch const bunch = makeProbeBunch(box, 1.0, center, center, center);
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {h, h, h};
        options.fftPadding = 2;
        options.marginCells = 4.0;
        picBunch::Field const field = picBunch::computeField(bunch, options);
        auto const& g = field.grid;

        double const q = picBunch::kElementaryCharge;
        double const eFactor = picBunch::kCoulombConstant;
        double const bFactor = picBunch::kCoulombConstant / picBunch::kSpeedOfLight;
        std::size_t const pid = static_cast<std::size_t>(
            std::llround((center - g.origin[0]) / h));

        if(haveCsv)
            std::fprintf(csv, "d,rel_error,l2_abs_error,scale,max_abs_error\n");

        double worst = 0.0;
        double worstAbs = 0.0;
        int worstAbsComponent = -1;
        char const* const componentName[6] = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
        std::fprintf(stdout, "# Coulomb+TSC direct-stencil check (u=0, padding=2)\n");
        std::fprintf(stdout, "d,rel_error,l2_abs_error,scale,max_abs_error\n");
        for(int d : {8, 16, 24, 32})
        {
            for(int a = 0; a < 3; ++a)
                for(int sign = -1; sign <= 1; sign += 2)
                {
                    std::array<std::size_t, 3> idx{pid, pid, pid};
                    idx[a] = static_cast<std::size_t>(static_cast<long>(pid) + static_cast<long>(sign * d));
                    auto pos = [&](int axisIndex) {
                        return g.origin[axisIndex] + static_cast<double>(idx[axisIndex]) * h;
                    };
                    double const x = pos(0);
                    double const y = pos(1);
                    double const z = pos(2);
                    double numE = 0.0;
                    double denE = 0.0;
                    // elementwise max |actual - expected| over all six components
                    // at this point; the L2 residual above covers the E components only
                    double rowMaxAbs = 0.0;
                    int rowMaxAbsComponent = -1;
                    for(int c = 0; c < 6; ++c)
                    {
                        int used = 0;
                        auto const ref = directTscCoulomb(g, center, center, center, x, y, z, c, used);
                        std::size_t const flat = idx[0] + g.n[0] * (idx[1] + g.n[1] * idx[2]);
                        double const expected = q * (c < 3 ? eFactor : bFactor) * ref[c];
                        double const actual = field.data[c][flat];
                        double const delta = actual - expected;
                        if(c < 3)
                        {
                            numE += delta * delta;
                            denE += expected * expected;
                        }
                        double const absDelta = std::abs(delta);
                        if(absDelta > rowMaxAbs)
                        {
                            rowMaxAbs = absDelta;
                            rowMaxAbsComponent = c;
                        }
                    }
                    if(rowMaxAbs > worstAbs)
                    {
                        worstAbs = rowMaxAbs;
                        worstAbsComponent = rowMaxAbsComponent;
                    }
                    double const rel = std::sqrt(numE / std::max(denE, 1e-300));
                    worst = std::max(worst, rel);
                    std::fprintf(
                        stdout,
                        "%d,%.6e,%.6e,%.6e,%.6e\n",
                        d,
                        rel,
                        std::sqrt(numE),
                        std::sqrt(denE),
                        rowMaxAbs);
                    if(haveCsv)
                        std::fprintf(
                            csv,
                            "%d,%.6e,%.6e,%.6e,%.6e\n",
                            d,
                            rel,
                            std::sqrt(numE),
                            std::sqrt(denE),
                            rowMaxAbs);
                }
        }
        std::fprintf(stdout, "# COULOMB_MAX_REL=%.6e (threshold 1e-12)\n", worst);
        std::fprintf(
            stderr,
            "COULOMB_MAX_REL=%.6e max_abs=%.6e max_abs_component=%s %s\n",
            worst,
            worstAbs,
            worstAbsComponent >= 0 ? componentName[worstAbsComponent] : "none",
            worst < 1e-12 ? "PASS" : "FAIL");
        return worst < 1e-12 ? 0 : 3;
    }

    // ------------------------------------------------------------- d sweep

    int runDSweep(double gamma, std::FILE* csv, bool haveCsv)
    {
        double const box = 128.0;
        double const h = 1.0;
        double const center = box / 2.0;
        picBunch::Bunch const bunch = makeProbeBunch(box, gamma, center, center, center);
        picBunch::ComputeOptions options;
        options.tolerance = 1e-3;
        options.cellSize = {h, h, h};
        options.fftPadding = 2;
        options.marginCells = 4.0;
        picBunch::Field const field = picBunch::computeField(bunch, options);
        auto const& g = field.grid;

        double const q = picBunch::kElementaryCharge;
        double const eFactor = picBunch::kCoulombConstant;
        double const bFactor = picBunch::kCoulombConstant / picBunch::kSpeedOfLight;
        double const uz = std::sqrt(std::max(0.0, gamma * gamma - 1.0));
        std::size_t const pid = static_cast<std::size_t>(
            std::llround((center - g.origin[0]) / h));

        if(haveCsv)
            std::fprintf(csv, "gamma,d,worst_rel_E,worst_rel_B,worst_rel_all\n");
        std::fprintf(stdout, "# d-sweep gamma=%g aspect=1 padding=2 h=%g\n", gamma, h);
        std::fprintf(stdout, "gamma,d,worst_rel_E,worst_rel_B,worst_rel_all\n");

        double firstBelow = -1.0;
        for(int d = 2; d <= 48; ++d)
        {
            double worstE = 0.0;
            double worstB = 0.0;
            double worstAll = 0.0;
            for(int a = 0; a < 3; ++a)
                for(int sign = -1; sign <= 1; sign += 2)
                {
                    std::array<std::size_t, 3> idx{pid, pid, pid};
                    idx[a] = static_cast<std::size_t>(static_cast<long>(pid) + static_cast<long>(sign * d));
                    double const x = g.origin[0] + static_cast<double>(idx[0]) * h;
                    double const y = g.origin[1] + static_cast<double>(idx[1]) * h;
                    double const z = g.origin[2] + static_cast<double>(idx[2]) * h;
                    double numE = 0.0, denE = 0.0, numB = 0.0, denB = 0.0, numAll = 0.0, denAll = 0.0;
                    for(int c = 0; c < 6; ++c)
                    {
                        double const ox = picBunch::kYeeOffset[c][0] * h;
                        double const oy = picBunch::kYeeOffset[c][1] * h;
                        double const oz = picBunch::kYeeOffset[c][2] * h;
                        std::array<double, 6> reference{};
                        picBunch::pointFieldNormalized(
                            x + ox - center,
                            y + oy - center,
                            z + oz - center,
                            0.0,
                            0.0,
                            uz,
                            reference.data());
                        double const expected = q * (c < 3 ? eFactor : bFactor) * reference[c];
                        std::size_t const flat = idx[0] + g.n[0] * (idx[1] + g.n[1] * idx[2]);
                        double const actual = field.data[c][flat];
                        double const delta = actual - expected;
                        if(c < 3)
                        {
                            numE += delta * delta;
                            denE += expected * expected;
                        }
                        else
                        {
                            numB += delta * delta;
                            denB += expected * expected;
                        }
                        numAll += delta * delta;
                        denAll += expected * expected;
                    }
                    worstE = std::max(worstE, std::sqrt(numE / std::max(denE, 1e-300)));
                    worstB = std::max(worstB, std::sqrt(numB / std::max(denB, 1e-300)));
                    worstAll = std::max(worstAll, std::sqrt(numAll / std::max(denAll, 1e-300)));
                }
            if(firstBelow < 0.0 && worstAll < 3e-4)
                firstBelow = static_cast<double>(d);
            std::fprintf(stdout, "%.17g,%d,%.6e,%.6e,%.6e\n", gamma, d, worstE, worstB, worstAll);
            if(haveCsv)
                std::fprintf(csv, "%.17g,%d,%.6e,%.6e,%.6e\n", gamma, d, worstE, worstB, worstAll);
        }
        std::fprintf(
            stderr,
            "D_SWEEP gamma=%g first_d_below_3e-4=%s\n",
            gamma,
            firstBelow < 0.0 ? "none" : std::to_string(static_cast<int>(firstBelow)).c_str());
        return 0;
    }

    // ------------------------------------------------------------- CLI

    struct Arguments
    {
        std::size_t particles = 100000;
        double box = 20.0;
        double gamma = 10.0;
        double spacing = 1.0;
        unsigned padding = 2;
        int repeat = 9;
        int warmup = 2;
        std::string csv;
        std::string rawCsv;
        bool sweep = false;
        bool coulomb = false;
        bool dSweep = false;
    };

    void usage(std::FILE* out)
    {
        std::fprintf(
            out,
            "Usage: picBunchBench --particles N --box-cells L --gamma G --spacing h --repeat R [--csv out.csv]\n"
            "       picBunchBench --sweep [--repeat R] [--csv out.csv]\n"
            "       picBunchBench --coulomb [--csv out.csv]\n"
            "       picBunchBench --d-sweep [--csv out.csv]\n"
            "\n"
            "Options:\n"
            "  --particles N     macro particles (default 100000)\n"
            "  --box-cells L     physical box extent in reference cells (default 20)\n"
            "  --gamma G         Lorentz factor of the shared +z momentum (default 10)\n"
            "  --spacing h       node spacing / cell size (default 1)\n"
            "  --padding p       FFT zero-padding factor (default 2)\n"
            "  --repeat R        timed repeats, min/median/max (default 9)\n"
            "  --warmup W        untimed warm-up repeats (default 2)\n"
            "  --sweep           run the paper matrix A-D\n"
            "  --coulomb         u=0 mesh vs direct TSC Coulomb stencil\n"
            "  --d-sweep         mesh vs analytic point field, d=2..48, gamma 1 and 10\n"
            "  --csv PATH        also write CSV rows to PATH\n"
            "  --raw-csv PATH    write one row per warm-up/repeat (summarize.py input)\n"
            "  --help            print this help\n");
    }

    Arguments parseArguments(int argc, char** argv)
    {
        Arguments args;
        for(int i = 1; i < argc; ++i)
        {
            std::string const option = argv[i];
            auto value = [&](char const* name) -> std::string {
                if(i + 1 >= argc)
                    throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++i];
            };
            if(option == "--particles")
                args.particles = std::stoull(value("--particles"));
            else if(option == "--box-cells")
                args.box = std::stod(value("--box-cells"));
            else if(option == "--gamma")
                args.gamma = std::stod(value("--gamma"));
            else if(option == "--spacing")
                args.spacing = std::stod(value("--spacing"));
            else if(option == "--padding")
                args.padding = static_cast<unsigned>(std::stoul(value("--padding")));
            else if(option == "--repeat")
                args.repeat = std::stoi(value("--repeat"));
            else if(option == "--warmup")
                args.warmup = std::stoi(value("--warmup"));
            else if(option == "--csv")
                args.csv = value("--csv");
            else if(option == "--raw-csv")
                args.rawCsv = value("--raw-csv");
            else if(option == "--sweep")
                args.sweep = true;
            else if(option == "--coulomb")
                args.coulomb = true;
            else if(option == "--d-sweep")
                args.dSweep = true;
            else if(option == "--help" || option == "-h")
            {
                usage(stdout);
                std::exit(0);
            }
            else
                throw std::invalid_argument("unknown option '" + option + "'");
        }
        if(args.particles == 0)
            throw std::invalid_argument("--particles must be positive");
        if(!(args.box > 0.0) || !(args.spacing > 0.0) || !(args.gamma >= 1.0))
            throw std::invalid_argument("--box-cells, --spacing and --gamma must be positive");
        if(args.repeat < 1)
            throw std::invalid_argument("--repeat must be at least 1");
        if(args.warmup < 0)
            throw std::invalid_argument("--warmup must not be negative");
        if(args.padding < 1)
            throw std::invalid_argument("--padding must be at least 1");
        return args;
    }

    std::string rowId(std::size_t particles, double box, double gamma, double spacing, unsigned padding)
    {
        char buffer[160];
        std::snprintf(
            buffer,
            sizeof(buffer),
            "N%zu_b%g_g%g_h%g_p%u",
            particles,
            box,
            gamma,
            spacing,
            padding);
        return buffer;
    }

} // namespace

int main(int argc, char** argv)
{
    Arguments args;
    try
    {
        args = parseArguments(argc, argv);
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "picBunchBench: %s\n\n", error.what());
        usage(stderr);
        return 2;
    }

    std::FILE* csv = nullptr;
    std::FILE* rawCsv = nullptr;
    if(!args.csv.empty())
    {
        csv = std::fopen(args.csv.c_str(), "w");
        if(!csv)
        {
            std::fprintf(stderr, "picBunchBench: cannot open '%s'\n", args.csv.c_str());
            return 2;
        }
    }
    if(!args.rawCsv.empty())
    {
        rawCsv = std::fopen(args.rawCsv.c_str(), "w");
        if(!rawCsv)
        {
            std::fprintf(stderr, "picBunchBench: cannot open '%s'\n", args.rawCsv.c_str());
            return 2;
        }
        writeRawHeader(rawCsv);
    }

    int status = 0;
    try
    {
        if(args.coulomb)
            status = runCoulombCheck(csv, csv != nullptr);
        else if(args.dSweep)
        {
            runDSweep(1.0, csv, csv != nullptr);
            runDSweep(10.0, csv, csv != nullptr);
        }
        else if(args.sweep)
        {
            writeCsvHeader(stdout);
            if(csv)
                writeCsvHeader(csv);
            struct Config
            {
                std::size_t particles;
                double box;
                double gamma;
                double spacing;
            };
            // device-level warm-up: raise clocks before the first timed row
            (void) runConfiguration(2000000u, 20.0, 10.0, 1.0, args.padding, 1, 1);
            std::vector<Config> matrix;
            // A: particle scaling
            for(double n : {1e3, 1e4, 1e5, 1e6, 2e6})
                matrix.push_back({static_cast<std::size_t>(n), 20.0, 10.0, 1.0});
            // B: box scaling
            for(double l : {20.0, 50.0, 100.0})
                matrix.push_back({2000000u, l, 10.0, 1.0});
            // C: gamma scaling
            for(double g : {1.0, 10.0, 100.0})
                matrix.push_back({2000000u, 20.0, g, 1.0});
            // D: spacing / resolution
            for(double h : {1.0, 0.5, 0.25})
                matrix.push_back({2000000u, 20.0, 10.0, h});
            for(auto const& c : matrix)
            {
                std::string const id = rowId(c.particles, c.box, c.gamma, c.spacing, args.padding);
                Row const row = runConfiguration(
                    c.particles,
                    c.box,
                    c.gamma,
                    c.spacing,
                    args.padding,
                    args.repeat,
                    args.warmup,
                    rawCsv,
                    id);
                writeCsvRow(stdout, row);
                if(csv)
                    writeCsvRow(csv, row);
                std::fflush(stdout);
                if(csv)
                    std::fflush(csv);
            }
        }
        else
        {
            std::string const id
                = rowId(args.particles, args.box, args.gamma, args.spacing, args.padding);
            Row const row = runConfiguration(
                args.particles,
                args.box,
                args.gamma,
                args.spacing,
                args.padding,
                args.repeat,
                args.warmup,
                rawCsv,
                id);
            writeCsvHeader(stdout);
            writeCsvRow(stdout, row);
            if(csv)
            {
                writeCsvHeader(csv);
                writeCsvRow(csv, row);
            }
        }
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "picBunchBench: error: %s\n", error.what());
        status = 1;
    }

    if(csv)
        std::fclose(csv);
    if(rawCsv)
        std::fclose(rawCsv);
    return status;
}
