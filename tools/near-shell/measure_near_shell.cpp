// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Measurement driver for the exact near shell.  Runs the ordinary mesh and the
// tiled F = F_mesh(all) - F_mesh(near) + F_exact(near) on a synthetic bunch and
// writes both fields as raw little-endian doubles for an independent reference
// comparison (tools/near-shell/compare_reference.py).

#include "picBunch/Compute.hpp"
#include "picBunch/NearShell.hpp"
#include "picBunch/Types.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr double kElementaryCharge = 1.602176634e-19;
    constexpr double kElectronMass = 9.1093837015e-31;
    constexpr double kSpeedOfLight = 299792458.0;

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
        std::vector<double> origin{0.0, 0.0, 0.0};
        std::vector<double> cell{1.0, 1.0, 1.0};
        std::vector<double> u{0.0, 0.0, 0.0};
        std::vector<double> source{0.0, 0.0, 0.0};
        std::vector<std::size_t> grid{32, 32, 32};
        double radius = 3.0;
        std::size_t tile = 4;
        std::string shell = "exact";
        int fastOrder = 5;
        std::size_t sources = 1;
        double cloudCells = 0.0;
        std::string prefix = "near_shell";

        for(int i = 1; i < argc; ++i)
        {
            std::string const a = argv[i];
            auto next = [&]() -> char const* {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if(a == "--origin")
                origin = parseTriple(next());
            else if(a == "--cell")
                cell = parseTriple(next());
            else if(a == "--u")
                u = parseTriple(next());
            else if(a == "--source")
                source = parseTriple(next());
            else if(a == "--grid")
            {
                auto v = parseTriple(next());
                grid = {static_cast<std::size_t>(v[0]), static_cast<std::size_t>(v[1]),
                        static_cast<std::size_t>(v[2])};
            }
            else if(a == "--radius")
                radius = std::stod(next());
            else if(a == "--tile")
                tile = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--shell")
                shell = next();
            else if(a == "--order")
                fastOrder = std::stoi(next());
            else if(a == "--sources")
                sources = static_cast<std::size_t>(std::stoul(next()));
            else if(a == "--cloud-cells")
                cloudCells = std::stod(next());
            else if(a == "--output")
                prefix = next();
            else
                throw std::runtime_error("unknown argument " + a);
        }

        picBunch::Bunch bunch;
        bunch.resize(sources);
        // Deterministic, seed-free LCG so the cost runs are reproducible.
        std::uint64_t state = 0x9e3779b97f4a7c15ull;
        auto nextUnit = [&state]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>((state >> 11) & ((1ull << 53) - 1))
                / static_cast<double>(1ull << 53);
        };
        for(std::size_t p = 0; p < sources; ++p)
        {
            double offset[3] = {0.0, 0.0, 0.0};
            if(cloudCells > 0.0)
                for(int a = 0; a < 3; ++a)
                    offset[a] = (nextUnit() - 0.5) * cloudCells * cell[a];
            bunch.x[p] = source[0] + offset[0];
            bunch.y[p] = source[1] + offset[1];
            bunch.z[p] = source[2] + offset[2];
            bunch.ux[p] = u[0];
            bunch.uy[p] = u[1];
            bunch.uz[p] = u[2];
            bunch.weight[p] = 1.0 / static_cast<double>(sources);
            bunch.charge[p] = -kElementaryCharge;
        }

        picBunch::ComputeOptions options;
        options.cellSize = {cell[0], cell[1], cell[2]};
        options.gridSize = {grid[0], grid[1], grid[2]};
        options.gridOrigin = {origin[0], origin[1], origin[2]};
        options.fftPadding = 2u;
        options.deterministicDeposit = true;

        auto t0 = std::chrono::steady_clock::now();
        picBunch::Field const mesh = picBunch::computeField(bunch, options);
        auto t1 = std::chrono::steady_clock::now();

        if(shell != "exact" && shell != "fast")
            throw std::runtime_error("--shell must be 'exact' or 'fast'");
        picBunch::NearShellOptions ns;
        ns.mode = (shell == "fast") ? picBunch::NearShellOptions::Mode::Fast
                                    : picBunch::NearShellOptions::Mode::Exact;
        ns.fastOrder = fastOrder;
        ns.radiusCells = radius;
        ns.tileNodes = tile;
        std::size_t pairs = 0;
        ns.exactPairs = &pairs;
        picBunch::Field const total = picBunch::computeFieldNearExact(bunch, options, ns);
        auto t2 = std::chrono::steady_clock::now();

        double const meshS = std::chrono::duration<double>(t1 - t0).count();
        double const totalS = std::chrono::duration<double>(t2 - t1).count();

        writeField(prefix + "_mesh.bin", mesh);
        writeField(prefix + "_total.bin", total);
        std::ofstream meta(prefix + "_meta.txt");
        meta.precision(17);
        meta << "grid " << grid[0] << " " << grid[1] << " " << grid[2] << "\n";
        meta << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
        meta << "cell " << cell[0] << " " << cell[1] << " " << cell[2] << "\n";
        meta << "u " << u[0] << " " << u[1] << " " << u[2] << "\n";
        meta << "source " << source[0] << " " << source[1] << " " << source[2] << "\n";
        meta << "radius " << radius << "\n";
        meta << "tile " << tile << "\n";
        meta << "shell " << shell << "\n";
        meta << "order " << fastOrder << "\n";
        meta << "sources " << sources << "\n";
        meta << "cloud_cells " << cloudCells << "\n";
        meta << "pairs " << pairs << "\n";
        meta << "mesh_seconds " << meshS << "\n";
        meta << "near_shell_seconds " << totalS << "\n";
        std::printf("pairs=%zu mesh=%.6f s near=%.6f s\n", pairs, meshS, totalS);
        return 0;
    }
    catch(std::exception const& e)
    {
        std::fprintf(stderr, "measure_near_shell: %s\n", e.what());
        return 1;
    }
}
