// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Loads the committed real-bunch fixture and derives the velocity binning
// (uniform or quantile edges) at each requested box count.  Prints the beam
// frame, the per-axis boundaries, the model numbers and the per-box
// charge-weighted angular statistics the cloud-reference comparison needs.
//
// This is a reference driver only: it does not compute the field.  The cloud
// reference (tools/velocity-bins/cloud_reference.py) consumes the representative
// velocities and the per-particle assignment to build the binned field and
// compare it against the per-particle cloud field at the in-bunch probe nodes.

#include "picBunch/VelocityBinning.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
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
} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string bunchPath;
        std::string edges = "uniform";
        std::vector<std::size_t> bins = {1, 2, 4, 8, 16};
        for(int i = 1; i < argc; ++i)
        {
            std::string const a = argv[i];
            auto next = [&]() -> char const* {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if(a == "--bunch")
                bunchPath = next();
            else if(a == "--bin-edges")
                edges = next();
            else if(a == "--bins")
                bins = parseSizeList(next());
            else
                throw std::runtime_error("unknown argument " + a);
        }
        if(bunchPath.empty())
            throw std::runtime_error("--bunch is required");
        if(edges != "uniform" && edges != "quantile")
            throw std::runtime_error("--bin-edges must be uniform or quantile");

        std::ifstream f(bunchPath);
        if(!f)
            throw std::runtime_error("cannot open " + bunchPath);
        std::string line;
        std::getline(f, line); // header
        picBunch::Bunch bunch;
        double xv, yv, zv, uvx, uvy, uvz, wv, cv;
        while(f >> xv >> yv >> zv >> uvx >> uvy >> uvz >> wv >> cv)
        {
            std::size_t const n = bunch.size();
            bunch.resize(n + 1u);
            bunch.x[n] = xv;
            bunch.y[n] = yv;
            bunch.z[n] = zv;
            bunch.ux[n] = uvx;
            bunch.uy[n] = uvy;
            bunch.uz[n] = uvz;
            bunch.weight[n] = wv;
            bunch.charge[n] = cv;
        }
        std::printf("loaded %zu particles from %s (edges=%s)\n", bunch.size(),
                    bunchPath.c_str(), edges.c_str());

        picBunch::VelocityBinningOptions policy;
        policy.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        policy.edgeRule
            = (edges == "quantile") ? picBunch::VelocityBinning::EdgeRule::Quantile
                                    : picBunch::VelocityBinning::EdgeRule::Uniform;
        for(std::size_t const B : bins)
        {
            policy.explicitBins = B;
            auto const t0 = std::chrono::steady_clock::now();
            picBunch::VelocityBinning const binning
                = picBunch::deriveVelocityBinning(bunch, 1.0e-3, policy);
            auto const t1 = std::chrono::steady_clock::now();
            double const wallS = std::chrono::duration<double>(t1 - t0).count();
            std::printf(
                "B=%zu per_axis=%zux%zux%zu edge_rule=%s model_sum=%.6e model_error=%.6e "
                "wall_s=%.6f\n",
                binning.count, binning.perAxis[0], binning.perAxis[1], binning.perAxis[2],
                (binning.edgeRule == picBunch::VelocityBinning::EdgeRule::Quantile)
                    ? "quantile"
                    : "uniform",
                binning.modelSum, binning.modelError, wallS);
            std::printf("  u_min  %.10e %.10e %.10e\n", binning.uMin[0], binning.uMin[1],
                        binning.uMin[2]);
            std::printf("  u_max  %.10e %.10e %.10e\n", binning.uMax[0], binning.uMax[1],
                        binning.uMax[2]);
            std::printf("  basis  ");
            for(int a = 0; a < 3; ++a)
                std::printf("%.10e %.10e %.10e  ", binning.basis[a][0], binning.basis[a][1],
                            binning.basis[a][2]);
            std::printf("\n");
            std::printf("  boundaries ");
            for(double const v : binning.axisBoundaries)
                std::printf("%.10e ", v);
            std::printf("\n");
            // The cloud-reference comparison consumes the representative
            // velocities and the per-particle assignment.  Write them to a
            // companion file so the stdout stays compact (the box_of line
            // for a 50k-particle bunch is ~200 KB).
            std::ofstream repFile(
                "rep_B" + std::to_string(binning.count) + "_edges_" + edges + ".txt");
            repFile.precision(17);
            repFile << "# rep_b " << binning.count << " (lab-frame u per box)  boxOf: one index per line\n";
            for(auto const& u : binning.representativeU)
                repFile << "rep_b " << u[0] << ' ' << u[1] << ' ' << u[2] << '\n';
            for(std::size_t const b : binning.boxOf)
                repFile << b << '\n';
        }
        return 0;
    }
    catch(std::exception const& e)
    {
        std::fprintf(stderr, "load_fixture: %s\n", e.what());
        return 1;
    }
}
