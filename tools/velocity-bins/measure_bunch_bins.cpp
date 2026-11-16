// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Calibration driver: load a real openPMD bunch (default species), run the
// binned mesh pipeline for a list of box counts (or the auto policy) on the
// caller's grid, and write the fields plus the metadata
// compare_velocity_bins.py needs (bunch file with unit-converted momenta,
// per-box model sums and representatives).  Pure diagnostic; not part of the
// shipped target set.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/VelocityBinning.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr double kElementaryCharge = 1.602176634e-19;

    std::vector<double> parseTriple(char const* text)
    {
        std::vector<double> out;
        std::string s(text);
        std::size_t pos = 0;
        while(true)
        {
            std::size_t next = s.find(',', pos);
            out.push_back(std::stod(s.substr(pos, next - pos)));
            if(next == std::string::npos)
                break;
            pos = next + 1;
        }
        while(out.size() < 3)
            out.push_back(out.back());
        return out;
    }

    void writeField(std::string const& path, picBunch::Field const& field)
    {
        std::ofstream f(path, std::ios::binary);
        if(!f)
            throw std::runtime_error("cannot open " + path);
        for(int c = 0; c < 6; ++c)
            f.write(reinterpret_cast<char const*>(field.data[c].data()),
                    static_cast<std::streamsize>(field.data[c].size() * sizeof(double)));
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string input;
        double electronMassKg = 9.1093837139e-31;
        double cLight = 299792458.0;
        double charge = -kElementaryCharge;
        bool originSet = false;
        std::vector<double> origin{0.0, 0.0, 0.0};
        bool cellSet = false;
        std::vector<double> cell{1.0e-6, 1.0e-6, 1.0e-6};
        std::vector<std::size_t> grid{48, 48, 48};
        std::vector<std::size_t> bins = {1, 2, 4, 8, 16};
        bool autoBins = false;
        double tolerance = 1.0e-3;
        std::size_t maxBins = picBunch::kDefaultMaxVelocityBins;
        bool perParticle = false;
        std::string prefix = "bunch_bins";
        std::string species;

        for(int i = 1; i < argc; ++i)
        {
            std::string const a = argv[i];
            auto next = [&]() -> char const* {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if(a == "--input")
                input = next();
            else if(a == "--species")
                species = next();
            else if(a == "--origin")
            {
                origin = parseTriple(next());
                originSet = true;
            }
            else if(a == "--cell")
            {
                cell = parseTriple(next());
                cellSet = true;
            }
            else if(a == "--grid")
            {
                auto v = parseTriple(next());
                grid = {static_cast<std::size_t>(v[0]), static_cast<std::size_t>(v[1]),
                        static_cast<std::size_t>(v[2])};
            }
            else if(a == "--bins")
            {
                std::string const text = next();
                if(text == "auto")
                {
                    autoBins = true;
                }
                else
                {
                    bins.clear();
                    std::string s(text);
                    std::size_t pos = 0;
                    while(pos < s.size())
                    {
                        std::size_t nxt = s.find(',', pos);
                        double const v = std::stod(s.substr(pos, nxt - pos));
                        bins.push_back(static_cast<std::size_t>(v));
                        if(nxt == std::string::npos)
                            break;
                        pos = nxt + 1;
                    }
                }
            }
            else if(a == "--tolerance")
                tolerance = std::stod(next());
            else if(a == "--max-bins")
                maxBins = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--per-particle")
                perParticle = true;
            else if(a == "--electron-mass")
                electronMassKg = std::stod(next());
            else if(a == "--c")
                cLight = std::stod(next());
            else if(a == "--charge")
                charge = std::stod(next());
            else if(a == "--output")
                prefix = next();
            else
                throw std::runtime_error("unknown argument " + a);
        }
        if(input.empty())
            throw std::runtime_error("--input is required");

        picBunch::Bunch bunch;
        {
            picBunch::BunchSelection sel;
            if(!species.empty())
                sel.speciesName = species;
            picBunch::BunchData const data = picBunch::readBunch(input, sel);
            bunch = data.bunch;
        }
        if(bunch.size() == 0u)
            throw std::runtime_error("loaded bunch is empty");

        // Unit-convert the loaded u (dimensionless p/(m c) is expected when the
        // reader already applied unitSI; the CLI does this conversion for us).
        // Here the reader stores u = p*unitSI/(m*c) only if base_mass/base_charge
        // were given; with none, u = p*unitSI (raw).  Detect: if the mean
        // |u| is not O(1)..O(1e4) the reader left raw values, so convert.
        double uMean = 0.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
            uMean += std::sqrt(bunch.ux[p] * bunch.ux[p] + bunch.uy[p] * bunch.uy[p]
                               + bunch.uz[p] * bunch.uz[p]);
        uMean /= static_cast<double>(bunch.size());
        double scale = 1.0;
        if(uMean > 1.0e5 || uMean < 1.0e-2)
            scale = 1.0 / (electronMassKg * cLight);
        if(scale != 1.0)
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                bunch.ux[p] *= scale;
                bunch.uy[p] *= scale;
                bunch.uz[p] *= scale;
            }
        std::printf("loaded %zu particles, mean |u| = %.6g (scale %.3e)\n", bunch.size(), uMean, scale);
        for(std::size_t p = 0; p < bunch.size(); ++p)
            bunch.charge[p] = charge;

        picBunch::ComputeOptions options;
        options.tolerance = tolerance;
        options.cellSize = {cell[0], cell[1], cell[2]};
        options.fftPadding = 2;
        options.velocityBinning.maxBins = maxBins;
        if(!cellSet)
        {
            std::fprintf(stderr, "measure_bunch_bins: --cell not set; cellSize=%g %g %g\n",
                        options.cellSize[0], options.cellSize[1], options.cellSize[2]);
            throw std::runtime_error("--cell is required");
        }
        if(originSet)
        {
            options.gridSize = {grid[0], grid[1], grid[2]};
            options.gridOrigin = {origin[0], origin[1], origin[2]};
        }

        // A true per-particle reference: one box per particle, each box
        // representative exactly the particle's own u.  This is the field the
        // binning error is measured against.  Computed before the binned runs
        // so it can be written as the reference field.
        if(perParticle)
        {
            picBunch::VelocityBinning pp;
            std::size_t const N = bunch.size();
            pp.derived = true;
            pp.multiVelocity = true;
            pp.count = N;
            pp.perAxis = {N, 1u, 1u};
            // Identity basis (lab frame) so boxIndex uses the raw u coords.
            pp.basis = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
            // One box per particle: split the x-axis range into N slabs so that
            // boxIndex floors each particle into a distinct box (a particle at
            // the exact upper edge clamps to the last box).
            double lo = bunch.ux[0], hi = bunch.ux[0];
            for(std::size_t p = 0; p < N; ++p)
            {
                lo = std::min(lo, bunch.ux[p]);
                hi = std::max(hi, bunch.ux[p]);
            }
            pp.uMin = {lo, 0.0, 0.0};
            pp.uMax = {hi, 0.0, 0.0};
            pp.representativeU.resize(N);
            pp.boxOf.resize(N);
            for(std::size_t p = 0; p < N; ++p)
            {
                pp.representativeU[p] = {bunch.ux[p], bunch.uy[p], bunch.uz[p]};
                pp.boxOf[p] = p;
            }
            pp.modelSum = 0.0;
            pp.modelConstant = 1.0;
            pp.modelError = 0.0;
            options.fixedBinning = pp;
            picBunch::StageTimes times;
            auto const t0 = std::chrono::steady_clock::now();
            picBunch::Field const field = picBunch::computeField(bunch, options);
            auto const t1 = std::chrono::steady_clock::now();
            double const wallS = std::chrono::duration<double>(t1 - t0).count();
            writeField(prefix + "_field_pp.bin", field);
            std::printf("per-particle reference written (%zu boxes), wall=%.2f s\n", N, wallS);
            std::exit(0);
        }

        std::ofstream bunchFile(prefix + "_bunch.txt");
        bunchFile.precision(17);
        bunchFile << "# x y z ux uy uz weight charge boxOf\n";

        std::vector<std::size_t> binList = bins;
        if(autoBins)
            binList = {0};

        std::ofstream report(prefix + "_report.txt");
        report.precision(17);
        bool bunchWritten = false;
        for(std::size_t const requested : binList)
        {
            options.velocityBinning.mode
                = autoBins ? picBunch::VelocityBinningOptions::Mode::Auto
                           : picBunch::VelocityBinningOptions::Mode::Explicit;
            options.velocityBinning.explicitBins = std::max<std::size_t>(1u, requested);

            picBunch::StageTimes times;
            picBunch::VelocityBinning binning;
            auto const t0 = std::chrono::steady_clock::now();
            picBunch::Field const field = picBunch::computeField(bunch, options, &times, &binning);
            auto const t1 = std::chrono::steady_clock::now();
            double const wallS = std::chrono::duration<double>(t1 - t0).count();

            std::size_t const B = binning.count;
            writeField(prefix + "_field_B" + std::to_string(B) + ".bin", field);

            std::vector<std::size_t> const boxOf = binning.boxOf;
            std::ofstream meta(prefix + "_meta_B" + std::to_string(B) + ".txt");
            meta.precision(17);
            meta << "grid " << grid[0] << " " << grid[1] << " " << grid[2] << "\n";
            meta << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
            meta << "cell " << cell[0] << " " << cell[1] << " " << cell[2] << "\n";
            meta << "gamma " << 0.0 << "\n";
            meta << "energy_spread " << 0.0 << "\n";
            meta << "angle_degrees " << 0.0 << "\n";
            meta << "sources " << bunch.size() << "\n";
            meta << "cloud_cells " << 0.0 << "\n";
            meta << "seed " << 0u << "\n";
            meta << "tolerance " << tolerance << "\n";
            meta << "fft_padding " << 2 << "\n";
            meta << "near_shell " << "none" << "\n";
            meta << "near_shell_radius " << 0.0 << "\n";
            meta << "bin_model_constant " << binning.modelConstant << "\n";
            meta << "max_bins " << maxBins << "\n";
            meta << "requested_bins " << requested << "\n";
            meta << "bins_used " << B << "\n";
            meta << "per_axis " << binning.perAxis[0] << " " << binning.perAxis[1] << " "
                 << binning.perAxis[2] << "\n";
            meta << "multi_velocity " << (binning.multiVelocity ? 1 : 0) << "\n";
            meta << "model_sum " << binning.modelSum << "\n";
            meta << "model_error " << binning.modelError << "\n";
            meta << "wall_seconds " << wallS << "\n";
            meta << "deposit_ms " << times.depositMs << "\n";
            meta << "kernel_ms " << times.kernelMs << "\n";
            meta << "fft_ms " << times.fftMs << "\n";
            meta << "readout_ms " << times.readoutMs << "\n";
            meta << "basis";
            for(auto const& row : binning.basis)
                meta << " " << row[0] << " " << row[1] << " " << row[2];
            meta << "\n";
            meta << "u_min " << binning.uMin[0] << " " << binning.uMin[1] << " "
                 << binning.uMin[2] << "\n";
            meta << "u_max " << binning.uMax[0] << " " << binning.uMax[1] << " "
                 << binning.uMax[2] << "\n";
            meta << "representative_u";
            for(auto const& u : binning.representativeU)
                meta << " " << u[0] << " " << u[1] << " " << u[2];
            meta << "\n";
            meta << "box_of";
            for(std::size_t const b : boxOf)
                meta << " " << b;
            meta << "\n";

            if(!bunchWritten)
            {
                bunchWritten = true;
                for(std::size_t p = 0; p < bunch.size(); ++p)
                    bunchFile << bunch.x[p] << ' ' << bunch.y[p] << ' ' << bunch.z[p] << ' '
                              << bunch.ux[p] << ' ' << bunch.uy[p] << ' ' << bunch.uz[p] << ' '
                              << bunch.weight[p] << ' ' << bunch.charge[p] << ' ' << boxOf[p] << '\n';
            }

            report << "bins_requested=" << requested << " bins_used=" << B
                   << " per_axis=" << binning.perAxis[0] << "x" << binning.perAxis[1] << "x"
                   << binning.perAxis[2] << " model_sum=" << binning.modelSum
                   << " model_error=" << binning.modelError << " wall_s=" << wallS << "\n";
            std::printf(
                "B=%zu (%zux%zux%zu) model_sum=%.6e model_error=%.6e wall=%.4f s "
                "(deposit %.3f ms, kernel %.3f ms, fft %.3f ms, readout %.3f ms)\n",
                B, binning.perAxis[0], binning.perAxis[1], binning.perAxis[2],
                binning.modelSum, binning.modelError, wallS, times.depositMs, times.kernelMs,
                times.fftMs, times.readoutMs);
        }
        return 0;
    }
    catch(std::exception const& e)
    {
        std::fprintf(stderr, "measure_bunch_bins: %s\n", e.what());
        return 1;
    }
}
