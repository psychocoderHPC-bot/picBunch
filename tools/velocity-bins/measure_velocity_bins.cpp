// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Measurement driver for velocity binning.  Generates a synthetic
// mixed-velocity bunch (Gaussian energy spread and/or angular spread about a
// beam axis), runs the binned mesh pipeline for a list of box counts (or the
// auto policy), and writes the fields plus all metadata the continuum
// reference comparison needs (tools/velocity-bins/compare_velocity_bins.py).
//
// Outputs (per requested box count B):
//   <prefix>_bunch.txt        x y z ux uy uz weight charge boxOf  (one line per
//                             particle; boxOf from the B-box run)
//   <prefix>_field_B<B>.bin   6 x n0*n1*n2 little-endian doubles, x fastest
//   <prefix>_meta_B<B>.txt    grid, bunch, binning and timing metadata
// The bunch file is written once (its boxOf column is that of the first run);
// the per-run boxOf also lives in the meta file.

#include "picBunch/Compute.hpp"
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

    std::vector<std::size_t> parseSizeList(char const* text)
    {
        std::vector<std::size_t> out;
        std::string s(text);
        std::size_t pos = 0;
        while(pos < s.size())
        {
            std::size_t next = s.find(',', pos);
            out.push_back(std::stoul(s.substr(pos, next - pos)));
            if(next == std::string::npos)
                break;
            pos = next + 1;
        }
        return out;
    }

    //! Deterministic seed-free LCG so the runs are reproducible.
    struct Rng
    {
        std::uint64_t state;

        explicit Rng(std::uint64_t seed)
            : state(seed == 0 ? 0x9e3779b97f4a7c15ull : seed)
        {
        }

        double unit()
        {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>((state >> 11) & ((1ull << 53) - 1))
                / static_cast<double>(1ull << 53);
        }

        //! Standard normal, Box-Muller (cached second draw).
        double gaussian()
        {
            if(spare)
            {
                spare = false;
                return spareValue;
            }
            double u1 = 0.0, u2 = 0.0;
            do
            {
                u1 = unit();
            } while(u1 <= 1.0e-300);
            u2 = unit();
            double const r = std::sqrt(-2.0 * std::log(u1));
            spareValue = r * std::sin(2.0 * M_PI * u2);
            spare = true;
            return r * std::cos(2.0 * M_PI * u2);
        }

    private:
        bool spare = false;
        double spareValue = 0.0;
    };

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
        double gamma = 10.0;
        double energySpread = 0.05; //!< sigma_gamma / gamma; 0 = none
        double angleDegrees = 0.0; //!< angular sigma in degrees; 0 = none
        std::vector<double> origin{0.0, 0.0, 0.0};
        std::vector<double> cell{1.0e-6, 1.0e-6, 1.0e-6};
        std::vector<std::size_t> grid{48, 48, 48};
        std::size_t sources = 512;
        double cloudCells = 3.0; //!< position cloud sigma in cells
        std::uint64_t seed = 0x9e3779b97f4a7c15ull;
        std::vector<std::size_t> bins = {1, 2, 4, 8, 16};
        bool autoBins = false;
        double tolerance = 1.0e-3;
        double modelConstant = picBunch::kDefaultVelocityModelConstant;
        std::size_t maxBins = picBunch::kDefaultMaxVelocityBins;
        std::size_t momentOrder = 0; //!< 0 = off; else run the moment path at this order
        unsigned fftPadding = 2u;
        bool deterministicDeposit = false;
        std::string nearShell = "none";
        double nearShellRadius = 0.0;
        std::size_t nearShellTile = 4;
        std::string prefix = "velocity_bins";

        for(int i = 1; i < argc; ++i)
        {
            std::string const a = argv[i];
            auto next = [&]() -> char const* {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if(a == "--gamma")
                gamma = std::stod(next());
            else if(a == "--energy-spread")
                energySpread = std::stod(next());
            else if(a == "--angle-degrees")
                angleDegrees = std::stod(next());
            else if(a == "--origin")
                origin = parseTriple(next());
            else if(a == "--cell")
                cell = parseTriple(next());
            else if(a == "--grid")
            {
                auto v = parseTriple(next());
                grid = {static_cast<std::size_t>(v[0]), static_cast<std::size_t>(v[1]),
                        static_cast<std::size_t>(v[2])};
            }
            else if(a == "--sources")
                sources = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--cloud-cells")
                cloudCells = std::stod(next());
            else if(a == "--seed")
                seed = std::stoull(next(), nullptr, 0);
            else if(a == "--bins")
            {
                std::string const text = next();
                if(text == "auto")
                    autoBins = true;
                else
                    bins = parseSizeList(text.c_str());
            }
            else if(a == "--tolerance")
                tolerance = std::stod(next());
            else if(a == "--bin-model-constant")
                modelConstant = std::stod(next());
            else if(a == "--max-bins")
                maxBins = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--moments")
            {
                std::size_t const v = static_cast<std::size_t>(std::stoul(next()));
                if(v > 3u)
                    throw std::runtime_error("--moments must be 0, 2 or 3");
                momentOrder = v;
            }
            else if(a == "--fft-padding")
                fftPadding = static_cast<unsigned>(std::stoul(next()));
            else if(a == "--deterministic-deposit")
                deterministicDeposit = true;
            else if(a == "--near-shell")
                nearShell = next();
            else if(a == "--near-shell-radius")
                nearShellRadius = std::stod(next());
            else if(a == "--near-shell-tile")
                nearShellTile = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--output")
                prefix = next();
            else
                throw std::runtime_error("unknown argument " + a);
        }
        if(!(gamma > 1.0) || !std::isfinite(gamma))
            throw std::runtime_error("--gamma must be finite and > 1");
        if(!(energySpread >= 0.0) || !std::isfinite(energySpread))
            throw std::runtime_error("--energy-spread must be finite and >= 0");
        if(!(angleDegrees >= 0.0) || !std::isfinite(angleDegrees))
            throw std::runtime_error("--angle-degrees must be finite and >= 0");
        if(sources == 0u)
            throw std::runtime_error("--sources must be positive");
        if(nearShell != "none" && nearShell != "exact" && nearShell != "fast")
            throw std::runtime_error("--near-shell must be none, exact or fast");
        if(!(nearShellRadius >= 0.0) || !std::isfinite(nearShellRadius))
            throw std::runtime_error("--near-shell-radius must be finite and non-negative");
        if(nearShellTile == 0u)
            throw std::runtime_error("--near-shell-tile must be positive");

        // ------------------------------------------------------- bunch generation
        // Beam axis: +z.  Energy spread: gamma_p = gamma*(1 + s*z1), clipped to
        // |z1| <= 4.  Angular spread: small-angle tilt (theta_x, theta_y) ~
        // N(0, sigma_theta) each, applied to the unit direction.
        Rng rng(seed);
        double const u0 = std::sqrt(gamma * gamma - 1.0);
        double const sigmaTheta = angleDegrees * M_PI / 180.0;
        picBunch::Bunch bunch;
        bunch.resize(sources);
        std::array<double, 3> const centre{
            origin[0] + 0.5 * static_cast<double>(grid[0]) * cell[0],
            origin[1] + 0.5 * static_cast<double>(grid[1]) * cell[1],
            origin[2] + 0.5 * static_cast<double>(grid[2]) * cell[2]};
        for(std::size_t p = 0; p < sources; ++p)
        {
            double z1 = 0.0, z2 = 0.0, z3 = 0.0;
            double pos[3], zpos[3];
            for(int a = 0; a < 3; ++a)
            {
                zpos[a] = rng.gaussian();
                pos[a] = centre[a] + cloudCells * cell[a] * zpos[a];
            }
            if(energySpread > 0.0)
                z1 = rng.gaussian();
            if(angleDegrees > 0.0)
            {
                z2 = rng.gaussian();
                z3 = rng.gaussian();
            }
            z1 = std::max(-4.0, std::min(4.0, z1));
            double const gammaP = (energySpread > 0.0) ? gamma * (1.0 + energySpread * z1) : gamma;
            double const umag = std::sqrt(std::max(1.0e-24, gammaP * gammaP - 1.0));
            double ux = 0.0, uy = 0.0;
            if(angleDegrees > 0.0)
            {
                double const tx = sigmaTheta * z2;
                double const ty = sigmaTheta * z3;
                double const inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
                ux = umag * tx * inv;
                uy = umag * ty * inv;
            }
            double const uz = umag * (angleDegrees > 0.0
                ? 1.0 / std::sqrt(1.0 + (sigmaTheta * z2) * (sigmaTheta * z2)
                                    + (sigmaTheta * z3) * (sigmaTheta * z3))
                : 1.0);
            bunch.x[p] = pos[0];
            bunch.y[p] = pos[1];
            bunch.z[p] = pos[2];
            bunch.ux[p] = ux;
            bunch.uy[p] = uy;
            bunch.uz[p] = uz;
            bunch.weight[p] = 1.0;
            bunch.charge[p] = -kElementaryCharge;
        }

        picBunch::ComputeOptions options;
        options.tolerance = tolerance;
        options.cellSize = {cell[0], cell[1], cell[2]};
        options.gridSize = {grid[0], grid[1], grid[2]};
        options.gridOrigin = {origin[0], origin[1], origin[2]};
        options.fftPadding = fftPadding;
        options.deterministicDeposit = deterministicDeposit;
        options.nearShell.mode = (nearShell == "fast")
            ? picBunch::NearShellOptions::Mode::Fast
            : (nearShell == "exact") ? picBunch::NearShellOptions::Mode::Exact
                                     : picBunch::NearShellOptions::Mode::None;
        options.nearShell.radiusCells = nearShellRadius;
        options.nearShell.tileNodes = nearShellTile;
        options.velocityBinning.modelConstant = modelConstant;
        options.velocityBinning.maxBins = maxBins;

        std::ofstream bunchFile(prefix + "_bunch.txt");
        if(!bunchFile)
            throw std::runtime_error("cannot open " + prefix + "_bunch.txt");
        bunchFile.precision(17);
        bunchFile << "# x y z ux uy uz weight charge boxOf\n";

        std::vector<std::size_t> binList = bins;
        if(autoBins)
            binList = {0}; //!< 0 = auto policy, resolved by the run

        std::ofstream report(prefix + "_report.txt");
        report.precision(17);
        picBunch::Field refField;
        std::size_t refB = 0u;
        for(std::size_t const requested : binList)
        {
            options.velocityBinning.mode
                = autoBins ? picBunch::VelocityBinningOptions::Mode::Auto
                           : picBunch::VelocityBinningOptions::Mode::Explicit;
            options.velocityBinning.explicitBins = std::max<std::size_t>(1u, requested);

            picBunch::StageTimes times;
            picBunch::VelocityBinning binning;
            auto const t0 = std::chrono::steady_clock::now();
            picBunch::Field const field
                = picBunch::computeField(bunch, options, &times, &binning);
            auto const t1 = std::chrono::steady_clock::now();
            double const wallS = std::chrono::duration<double>(t1 - t0).count();

            std::size_t const B = binning.count;
            if(B > refB)
            {
                refField = field;
                refB = B;
            }
            writeField(prefix + "_field_B" + std::to_string(B) + ".bin", field);

            std::vector<std::size_t> const boxOf = binning.boxOf;
            std::ofstream meta(prefix + "_meta_B" + std::to_string(B) + ".txt");
            meta.precision(17);
            meta << "grid " << grid[0] << " " << grid[1] << " " << grid[2] << "\n";
            meta << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
            meta << "cell " << cell[0] << " " << cell[1] << " " << cell[2] << "\n";
            meta << "gamma " << gamma << "\n";
            meta << "energy_spread " << energySpread << "\n";
            meta << "angle_degrees " << angleDegrees << "\n";
            meta << "sources " << sources << "\n";
            meta << "cloud_cells " << cloudCells << "\n";
            meta << "seed " << seed << "\n";
            meta << "tolerance " << tolerance << "\n";
            meta << "fft_padding " << fftPadding << "\n";
            meta << "near_shell " << nearShell << "\n";
            meta << "near_shell_radius " << nearShellRadius << "\n";
            meta << "bin_model_constant " << modelConstant << "\n";
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

            // The bunch file is written once, with the first run's boxOf.
            static bool bunchWritten = false;
            if(!bunchWritten)
            {
                bunchWritten = true;
                for(std::size_t p = 0; p < sources; ++p)
                    bunchFile << bunch.x[p] << ' ' << bunch.y[p] << ' ' << bunch.z[p] << ' '
                              << bunch.ux[p] << ' ' << bunch.uy[p] << ' ' << bunch.uz[p] << ' '
                              << bunch.weight[p] << ' ' << bunch.charge[p] << ' ' << boxOf[p]
                              << '\n';
            }

            report << "bins_requested=" << requested << " bins_used=" << B
                   << " per_axis=" << binning.perAxis[0] << "x" << binning.perAxis[1] << "x"
                   << binning.perAxis[2] << " model_sum=" << binning.modelSum
                   << " model_error=" << binning.modelError << " wall_s=" << wallS
                   << " deposit_ms=" << times.depositMs << " kernel_ms=" << times.kernelMs
                   << " fft_ms=" << times.fftMs << " readout_ms=" << times.readoutMs << "\n";
            std::printf(
                "B=%zu (%zux%zux%zu) model_sum=%.6e model_error=%.6e wall=%.4f s "
                "(deposit %.3f ms, kernel %.3f ms, fft %.3f ms, readout %.3f ms)\n",
                B,
                binning.perAxis[0],
                binning.perAxis[1],
                binning.perAxis[2],
                binning.modelSum,
                binning.modelError,
                wallS,
                times.depositMs,
                times.kernelMs,
                times.fftMs,
                times.readoutMs);
        }

        if(momentOrder > 0u && refB > 0u)
        {
            options.velocityMoments.mode = picBunch::VelocityMomentsOptions::Mode::Explicit;
            options.velocityMoments.order = momentOrder;
            picBunch::StageTimes mtimes;
            picBunch::VelocityMomentsReport mrep{};
            auto const m0 = std::chrono::steady_clock::now();
            picBunch::Field const mfield
                = picBunch::computeField(bunch, options, &mtimes, nullptr, &mrep);
            auto const m1 = std::chrono::steady_clock::now();
            double const mwall = std::chrono::duration<double>(m1 - m0).count();
            double num = 0.0;
            double den = 0.0;
            for(int c = 0; c < 6; ++c)
                for(std::size_t i = 0; i < mfield.data[c].size(); ++i)
                {
                    double const d = mfield.data[c][i] - refField.data[c][i];
                    num += d * d;
                    den += refField.data[c][i] * refField.data[c][i];
                }
            double const relL2 = std::sqrt(num) / std::sqrt(den);
            report << "moment_order=" << momentOrder << " used=" << (mrep.used ? 1 : 0)
                   << " fell_back=" << (mrep.fellBack ? 1 : 0) << " probe=" << mrep.probe
                   << " rel_l2_vs_B" << refB << "=" << relL2 << " wall_s=" << mwall
                   << " deposit_ms=" << mtimes.depositMs << " kernel_ms=" << mtimes.kernelMs
                   << " fft_ms=" << mtimes.fftMs << " readout_ms=" << mtimes.readoutMs << "\n";
            std::printf(
                "moment order %zu: used=%d fell_back=%d rel_l2_vs_B%zu=%.4e wall=%.4f s "
                "(deposit %.3f ms, kernel %.3f ms, fft %.3f ms, readout %.3f ms)\n",
                momentOrder,
                mrep.used ? 1 : 0,
                mrep.fellBack ? 1 : 0,
                refB,
                relL2,
                mwall,
                mtimes.depositMs,
                mtimes.kernelMs,
                mtimes.fftMs,
                mtimes.readoutMs);
        }
        return 0;
    }
    catch(std::exception const& e)
    {
        std::fprintf(stderr, "measure_velocity_bins: %s\n", e.what());
        return 1;
    }
}
