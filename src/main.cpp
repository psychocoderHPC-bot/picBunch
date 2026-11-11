// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// picBunch command line: openPMD particles in, free-space E/B out.
// --tolerance sizes the FFT padding and the velocity bin count.  It is not an
// end-to-end error guarantee: near-source quadrature, far-tail and checkpoint
// precision have separate budgets (README, "Accuracy and error budget").

#include "picBunch/Compute.hpp"
#include "picBunch/Config.hpp"
#include "picBunch/FarTail.hpp"
#include "picBunch/Mpi.hpp"
#include "picBunch/NearShell.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Placement.hpp"
#include "picBunch/Types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using picBunch::Idx3;
    using picBunch::IVec3;
    using picBunch::UVec3;
    using picBunch::Vec3;

    constexpr char const* compiledBackend()
    {
#if defined(PICBUNCH_BACKEND_CUDA)
        return "cuda";
#elif defined(PICBUNCH_BACKEND_HIP)
        return "hip";
#else
        return "cpu";
#endif
    }

    void usage(std::FILE* out)
    {
        std::fprintf(
            out,
            "Usage:\n"
            "  picBunch --input <bunch.h5> --destination <checkpoint-dir> --config <params.toml>\n"
            "  picBunch --input <openPMD file/series> [advanced options]\n"
            "\n"
            "Placement interface (documented):\n"
            "  --config <path>       TOML configuration (see docs/configuration.md)\n"
            "  --destination <dir>   PIConGPU checkpoint directory to create\n"
            "\n"
            "Advanced options (legacy CLI; no unit flags in the documented usage):\n"
            "  --input <path>        openPMD particle series (required)\n"
            "  --species <name|idx>  particle species (default: first)\n"
            "  --output <path>       write E/B as an openPMD series\n"
            "  --input-iteration <N> input iteration index (default: first)\n"
            "  --output-iteration <N> output iteration index (default: 0)\n"
            "  --cell-size <dx>[,<dy>[,<dz>]]\n"
            "                        physical cell size in metres (overrides the\n"
            "                        input mesh; required when there is no mesh)\n"
            "  --grid <nx>[,<ny>[,<nz>]]  explicit computation/output node counts (even)\n"
            "  --origin <x>[,<y>[,<z>]]   computation grid origin in metres (overrides\n"
            "                        the input mesh gridGlobalOffset)\n"
            "  --global-grid <nx>[,<ny>[,<nz>]]\n"
            "                        written domain node counts (even); the computation\n"
            "                        stays on the padded bunch box and cells outside it\n"
            "                        are filled by the far-tail multipole (zero only with\n"
            "                        --allow-field-truncation)\n"
            "  --global-origin <x>[,<y>[,<z>]]\n"
            "                        written domain origin in metres (default: mesh\n"
            "                        gridGlobalOffset, else 0)\n"
            "  --allow-field-truncation\n"
            "                        keep the requested compute box even when it is\n"
            "                        smaller than the tolerance-derived field margin;\n"
            "                        writes unjustified zeros outside it (diagnostic)\n"
            "  --restart-format <name>    standard|picongpu (default: standard)\n"
            "  --output-dir <dir>    PIConGPU checkpoint directory (default: checkpoints)\n"
            "  --picongpu-unit-length <m>   length unit for the restart writer\n"
            "  --picongpu-unit-e <V/m>      E-field unit for the restart writer\n"
            "  --picongpu-unit-b <T>        B-field unit for the restart writer\n"
            "  --picongpu-step <N>          checkpoint step (default: output iteration)\n"
            "  --picongpu-species <name>    require this species in the input; every\n"
            "                               input species is copied to the checkpoint\n"
            "  --picongpu-dt <value>        stored dt attribute (default: 0)\n"
            "  --deterministic-deposit\n"
            "                        serial host TSC deposit (reproducible bitwise;\n"
            "                        slower, diagnostic only)\n"
            "  --tolerance <value>   relative accuracy target of the free-space\n"
            "                        convolution (default: 1e-3; floor 1e-14)\n"
            "  --fft-padding <N>     FFT zero-padding factor, an integer in [0,8]\n"
            "                        (default: 0 = tolerance policy).  1 disables\n"
            "                        padding (diagnostic: wrapped images leak into\n"
            "                        the output); values above 3 only add cost\n"
            "  --near-shell <mode>   none|exact|fast|fast-gpu (default: none).  exact\n"
            "                        replaces the mesh contribution of every source\n"
            "                        within the shell by the adaptive reference (a\n"
            "                        reference mode); fast uses the asinh box +\n"
            "                        subtracted-GL remainder, on the CUDA device when\n"
            "                        available and on the host otherwise; fast-gpu is\n"
            "                        the explicit device request and falls back to\n"
            "                        the host path when no device is present.\n"
            "  --near-shell-radius <cells>  shell radius in PIC cells (default: derived\n"
            "                        from --tolerance: 3 at 1e-3, +1 per decade, max 6)\n"
            "  --near-shell-tile <n> tile edge in nodes for the local subtraction FFT\n"
            "                        (default: 4)\n"
            "  --near-shell-fast-order <n>  Gauss order of the fast remainder (default: 5)\n"
            "  --near-shell-oblique-order <auto|N>\n"
            "                        oblique singular-box order: auto (default) raises it\n"
            "                        per tile from the source gamma, N in [4,12] forces it\n"
            "                        (4 is the fast path)\n"
            "  --near-shell-exact-cells <n>\n"
            "                        fast-path exact routing: -2 = automatic (default:\n"
            "                        observer cell at tolerance 1e-3, all in-support\n"
            "                        cells at tolerance <= 1e-4; gamma >= 100 always\n"
            "                        exact), -1 = legacy quadrature only, 0 = observer\n"
            "                        cell only, 1 = add face neighbours, 2 = all\n"
            "                        in-support cells\n"
            "  --velocity-bins <N>   velocity box count for a multi-velocity bunch;\n"
            "                        0 = auto: smallest count whose error-model\n"
            "                        estimate is below --tolerance (default: 0)\n"
            "  --bin-model-constant <C>\n"
            "                        calibration constant of the bin error model\n"
            "                        (diagnostic override of the shipped fit)\n"
            "  --max-velocity-bins <N>\n"
            "                        box-count cap of the auto policy (default: 256);\n"
            "                        an infeasible requirement fails with the needed\n"
            "                        count\n"
            "  --fft-mode <mode>     auto|replicated|distributed (default: auto).\n"
            "                        distributed decomposes the padded box by z-slabs\n"
            "                        and transposes with MPI_Alltoallv; auto selects it\n"
            "                        when the padded box exceeds the per-rank limit\n"
            "  --shape <shape>       cic|tsc|pqs particle shape for the charge deposit\n"
            "                        (default: tsc); must match the PIConGPU species shape\n"
            "  --velocity-moments <m>  off|auto|2|3 (default: off).  Replaces the\n"
            "                        per-velocity-box FFT chain with the velocity-moment\n"
            "                        spectral decomposition: '2'/'3' fix the Taylor order,\n"
            "                        'auto' probes order 3 in k-space.  Every order is\n"
            "                        validated by the same order-2-vs-3 probe; when it\n"
            "                        exceeds 0.05 the run warns and falls back to the\n"
            "                        binned path.  A non-finite moment field always falls\n"
            "                        back.  gamma-100 bunches are outside the validated\n"
            "                        regime (order 3 does not converge there)\n"
            "  --velocity-moments-allow-unconverged\n"
            "                        diagnostic: keep an explicit --velocity-moments 2|3\n"
            "                        even when its probe exceeds 0.05; the run warns and\n"
            "                        the report marks the field as unconverged\n"
            "  --multipole-group-cells <N>\n"
            "                        spatial supercell size for the far-tail multipoles,\n"
            "                        in PIC cells (default: 16; 0 = one global multipole)\n"
            "  --max-particle-chunk-gb <GB>\n"
            "                        stream the particle species in chunks of at most\n"
            "                        this many gigabytes (float; 0 = load all).  Requires\n"
            "                        explicit --grid, --origin, --cell-size and a fixed\n"
            "                        --velocity-bins N; --global-grid is not combined with\n"
            "                        chunking yet\n"
            "  --backend <name>      auto|cuda|hip|cpu (default: auto)\n"
            "  --help                print this help\n"
            "\n"
            "Compiled backend: %s\n",
            compiledBackend());
    }

    struct Arguments
    {
        std::string input;
        std::optional<std::string> destination;
        std::optional<std::string> config;
        std::optional<std::string> species;
        std::optional<std::string> output;
        std::optional<std::uint64_t> inputIteration;
        std::uint64_t outputIteration = 0;
        double tolerance = 1e-3;
        unsigned fftPadding = 0; //!< 0 = tolerance policy
        std::string nearShell = "none";
        double nearShellRadius = 0.0;
        std::size_t nearShellTile = 4;
        int nearShellFastOrder = 5;
        std::string nearShellObliqueOrder = "auto";
        int nearShellExactCells = -2;
        bool deterministicDeposit = false;
        bool allowFieldTruncation = false;
        std::string backend = "auto";
        std::string fftMode = "auto";
        std::string velocityMoments = "off"; //!< off|auto|2|3
        bool velocityMomentsAllowUnconverged = false; //!< diagnostic override
        std::size_t velocityBins = 0; //!< 0 = auto (tolerance-driven)
        double binModelConstant = picBunch::kDefaultVelocityModelConstant;
        std::size_t maxVelocityBins = picBunch::kDefaultMaxVelocityBins;
        std::optional<Vec3> cellSize;
        std::optional<Idx3> grid;
        std::optional<Vec3> origin;
        std::string shape = "tsc";
        int multipoleGroupCells = 16;
        double maxParticleChunkGb = 0.0;
        std::optional<Idx3> globalGrid;
        std::optional<Vec3> globalOrigin;
        std::string restartFormat = "standard";
        std::string outputDir = "checkpoints";
        std::optional<double> unitLength;
        std::optional<double> unitEField;
        std::optional<double> unitBField;
        std::optional<std::uint64_t> picongpuStep;
        double picongpuDt = 0.0;
        std::optional<std::string> picongpuSpecies;
    };

    bool isInteger(std::string const& text)
    {
        if(text.empty())
            return false;
        std::size_t start = (text[0] == '+') ? 1 : 0;
        if(start == text.size())
            return false;
        for(std::size_t i = start; i < text.size(); ++i)
            if(text[i] < '0' || text[i] > '9')
                return false;
        return true;
    }

    //! Strict non-negative integer parse: reject trailing junk such as "12abc".
    std::uint64_t parseIteration(std::string const& text, char const* option)
    {
        if(!isInteger(text))
            throw std::invalid_argument(
                std::string("invalid value '") + text + "' for " + option
                + ": expected a non-negative decimal integer");
        try
        {
            return std::stoull(text);
        }
        catch(std::exception const&)
        {
            throw std::invalid_argument(
                std::string("value '") + text + "' for " + option + " is out of range");
        }
    }

    //! Strict floating-point parse: reject trailing junk such as "1e-3abc".
    double parseDoubleStrict(std::string const& text, char const* option)
    {
        std::size_t consumed = 0;
        double value = 0.0;
        try
        {
            value = std::stod(text, &consumed);
        }
        catch(std::exception const&)
        {
            throw std::invalid_argument(
                std::string("invalid value '") + text + "' for " + option + ": expected a number");
        }
        if(consumed != text.size())
            throw std::invalid_argument(
                std::string("invalid value '") + text + "' for " + option + ": trailing characters");
        return value;
    }

    //! Strict floating-point parse: reject trailing junk such as "1e-3abc".
    double parseTolerance(std::string const& text)
    {
        return parseDoubleStrict(text, "--tolerance");
    }

    //! Split on commas; require either one value (repeated) or exactly three.
    std::vector<std::string> splitTriple(std::string const& text, char const* option)
    {
        std::vector<std::string> parts;
        std::string current;
        for(char c : text)
        {
            if(c == ',')
            {
                parts.push_back(current);
                current.clear();
            }
            else
                current += c;
        }
        parts.push_back(current);
        if(parts.size() != 1u && parts.size() != 3u)
            throw std::invalid_argument(
                std::string("invalid value '") + text + "' for " + option
                + ": expected one value or three comma-separated values");
        for(auto const& part : parts)
            if(part.empty())
                throw std::invalid_argument(
                    std::string("invalid value '") + text + "' for " + option + ": empty component");
        if(parts.size() == 1u)
            parts = {parts[0], parts[0], parts[0]};
        return parts;
    }

    Vec3 parseTripleDouble(std::string const& text, char const* option)
    {
        auto const parts = splitTriple(text, option);
        Vec3 result{};
        for(int a = 0; a < 3; ++a)
        {
            std::size_t consumed = 0;
            double value = 0.0;
            try
            {
                value = std::stod(parts[a], &consumed);
            }
            catch(std::exception const&)
            {
                throw std::invalid_argument(
                    std::string("invalid value '") + parts[a] + "' for " + option);
            }
            if(consumed != parts[a].size() || !std::isfinite(value))
                throw std::invalid_argument(
                    std::string("invalid value '") + parts[a] + "' for " + option);
            result[a] = value;
        }
        return result;
    }

    Idx3 parseTripleSize(std::string const& text, char const* option)
    {
        auto const parts = splitTriple(text, option);
        Idx3 result{};
        for(int a = 0; a < 3; ++a)
        {
            std::uint64_t const value = parseIteration(parts[a], option);
            if(value == 0u || value > 0xffffffffu)
                throw std::invalid_argument(
                    std::string("invalid grid size '") + parts[a] + "' for " + option);
            result[a] = static_cast<std::size_t>(value);
        }
        return result;
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
            if(option == "--input")
                args.input = value("--input");
            else if(option == "--destination")
                args.destination = value("--destination");
            else if(option == "--config")
                args.config = value("--config");
            else if(option == "--species")
                args.species = value("--species");
            else if(option == "--output")
                args.output = value("--output");
            else if(option == "--input-iteration")
                args.inputIteration = parseIteration(value("--input-iteration"), "--input-iteration");
            else if(option == "--output-iteration")
                args.outputIteration = parseIteration(value("--output-iteration"), "--output-iteration");
            else if(option == "--tolerance")
                args.tolerance = parseTolerance(value("--tolerance"));
            else if(option == "--fft-padding")
            {
                std::uint64_t const padding
                    = parseIteration(value("--fft-padding"), "--fft-padding");
                if(padding > 8u)
                    throw std::invalid_argument(
                        "--fft-padding must be between 0 and 8 (0 selects the tolerance policy)");
                args.fftPadding = static_cast<unsigned>(padding);
            }
            else if(option == "--near-shell")
                args.nearShell = value("--near-shell");
            else if(option == "--near-shell-radius")
                args.nearShellRadius
                    = parseTripleDouble(value("--near-shell-radius"), "--near-shell-radius")[0];
            else if(option == "--near-shell-tile")
                args.nearShellTile
                    = parseIteration(value("--near-shell-tile"), "--near-shell-tile");
            else if(option == "--near-shell-fast-order")
                args.nearShellFastOrder
                    = static_cast<int>(parseIteration(value("--near-shell-fast-order"), "--near-shell-fast-order"));
            else if(option == "--near-shell-oblique-order")
                args.nearShellObliqueOrder = value("--near-shell-oblique-order");
            else if(option == "--near-shell-exact-cells")
            {
                std::size_t used = 0;
                auto const text = value("--near-shell-exact-cells");
                int cells = 0;
                try
                {
                    cells = std::stoi(text, &used);
                }
                catch(std::exception const&)
                {
                    used = 0;
                }
                if(used != text.size() || cells < -2 || cells > 2)
                    throw std::invalid_argument(
                        "--near-shell-exact-cells must be an integer in [-2, 2]");
                args.nearShellExactCells = cells;
            }
            else if(option == "--velocity-bins")
                args.velocityBins = parseIteration(value("--velocity-bins"), "--velocity-bins");
            else if(option == "--bin-model-constant")
                args.binModelConstant
                    = parseDoubleStrict(value("--bin-model-constant"), "--bin-model-constant");
            else if(option == "--max-velocity-bins")
                args.maxVelocityBins
                    = parseIteration(value("--max-velocity-bins"), "--max-velocity-bins");
            else if(option == "--deterministic-deposit")
                args.deterministicDeposit = true;
            else if(option == "--allow-field-truncation")
                args.allowFieldTruncation = true;
            else if(option == "--shape")
                args.shape = value("--shape");
            else if(option == "--velocity-moments")
                args.velocityMoments = value("--velocity-moments");
            else if(option == "--velocity-moments-allow-unconverged")
                args.velocityMomentsAllowUnconverged = true;
            else if(option == "--multipole-group-cells")
            {
                args.multipoleGroupCells
                    = static_cast<int>(parseIteration(value("--multipole-group-cells"), "--multipole-group-cells"));
                if(args.multipoleGroupCells < 0)
                    throw std::invalid_argument("--multipole-group-cells must be >= 0");
            }
            else if(option == "--max-particle-chunk-gb")
            {
                args.maxParticleChunkGb = std::stod(value("--max-particle-chunk-gb"));
                if(!std::isfinite(args.maxParticleChunkGb) || args.maxParticleChunkGb < 0.0)
                    throw std::invalid_argument("--max-particle-chunk-gb must be finite and >= 0");
            }
            else if(option == "--backend")
                args.backend = value("--backend");
            else if(option == "--fft-mode")
                args.fftMode = value("--fft-mode");
            else if(option == "--cell-size")
                args.cellSize = parseTripleDouble(value("--cell-size"), "--cell-size");
            else if(option == "--grid")
                args.grid = parseTripleSize(value("--grid"), "--grid");
            else if(option == "--origin")
                args.origin = parseTripleDouble(value("--origin"), "--origin");
            else if(option == "--global-grid")
                args.globalGrid = parseTripleSize(value("--global-grid"), "--global-grid");
            else if(option == "--global-origin")
                args.globalOrigin = parseTripleDouble(value("--global-origin"), "--global-origin");
            else if(option == "--restart-format")
                args.restartFormat = value("--restart-format");
            else if(option == "--output-dir")
                args.outputDir = value("--output-dir");
            else if(option == "--picongpu-unit-length")
                args.unitLength = parseTripleDouble(value("--picongpu-unit-length"), "--picongpu-unit-length")[0];
            else if(option == "--picongpu-unit-e")
                args.unitEField = parseTripleDouble(value("--picongpu-unit-e"), "--picongpu-unit-e")[0];
            else if(option == "--picongpu-unit-b")
                args.unitBField = parseTripleDouble(value("--picongpu-unit-b"), "--picongpu-unit-b")[0];
            else if(option == "--picongpu-step")
                args.picongpuStep = parseIteration(value("--picongpu-step"), "--picongpu-step");
            else if(option == "--picongpu-species")
                args.picongpuSpecies = value("--picongpu-species");
            else if(option == "--picongpu-dt")
                args.picongpuDt = parseTripleDouble(value("--picongpu-dt"), "--picongpu-dt")[0];
            else if(option == "--help" || option == "-h")
            {
                usage(stdout);
                std::exit(0);
            }
            else
                throw std::invalid_argument("unknown option '" + option + "'");
        }
        if(args.input.empty())
            throw std::invalid_argument("--input is required");
        if(args.config && !args.destination)
            throw std::invalid_argument("--config requires --destination");
        if(args.destination && !args.config)
            throw std::invalid_argument("--destination requires --config");
        if(!(args.tolerance > 0.0) || !std::isfinite(args.tolerance))
            throw std::invalid_argument("--tolerance must be finite and positive");
        if(args.fftPadding > 8u)
            throw std::invalid_argument(
                "--fft-padding must be between 0 and 8 (0 selects the tolerance policy)");
        if(args.backend != "auto" && args.backend != compiledBackend())
            throw std::invalid_argument(
                "--backend " + args.backend + " is not available; this build uses " + compiledBackend());
        if(args.nearShell != "none" && args.nearShell != "exact" && args.nearShell != "fast"
           && args.nearShell != "fast-gpu")
            throw std::invalid_argument("--near-shell must be 'none', 'exact', 'fast' or 'fast-gpu'");
        if(args.shape != "cic" && args.shape != "tsc" && args.shape != "pqs")
            throw std::invalid_argument("--shape must be 'cic', 'tsc' or 'pqs'");
        if(args.velocityMoments != "off" && args.velocityMoments != "auto"
           && args.velocityMoments != "2" && args.velocityMoments != "3")
            throw std::invalid_argument("--velocity-moments must be 'off', 'auto', '2' or '3'");
        if(args.fftMode != "auto" && args.fftMode != "replicated" && args.fftMode != "distributed")
            throw std::invalid_argument("--fft-mode must be 'auto', 'replicated' or 'distributed'");
        if(!(args.nearShellRadius >= 0.0) || !std::isfinite(args.nearShellRadius))
            throw std::invalid_argument("--near-shell-radius must be finite and non-negative");
        if(args.nearShellTile == 0u)
            throw std::invalid_argument("--near-shell-tile must be positive");
        if(args.nearShellExactCells < -2 || args.nearShellExactCells > 2)
            throw std::invalid_argument("--near-shell-exact-cells must be in [-2, 2]");
        if(args.nearShellFastOrder < 2 || args.nearShellFastOrder > 32)
            throw std::invalid_argument("--near-shell-fast-order must be between 2 and 32");
        if(args.nearShellObliqueOrder != "auto")
        {
            int order = 0;
            try
            {
                std::size_t used = 0;
                order = std::stoi(args.nearShellObliqueOrder, &used);
                if(used != args.nearShellObliqueOrder.size())
                    throw std::invalid_argument("trailing characters");
            }
            catch(std::exception const&)
            {
                throw std::invalid_argument(
                    "--near-shell-oblique-order must be 'auto' or an integer in [4, 12]");
            }
            if(order < 4 || order > 12)
                throw std::invalid_argument(
                    "--near-shell-oblique-order must be 'auto' or an integer in [4, 12]");
        }
        if(args.maxVelocityBins < 1u)
            throw std::invalid_argument("--max-velocity-bins must be >= 1");
        if(!(args.binModelConstant > 0.0) || !std::isfinite(args.binModelConstant))
            throw std::invalid_argument("--bin-model-constant must be finite and positive");
        if(args.restartFormat != "standard" && args.restartFormat != "picongpu")
            throw std::invalid_argument("--restart-format must be 'standard' or 'picongpu'");
        if(args.restartFormat == "picongpu" && args.output)
            throw std::invalid_argument("--restart-format picongpu writes to --output-dir; do not pass --output");
        if(args.cellSize)
            for(std::size_t a = 0u; a < 3u; ++a)
                if(!((*args.cellSize)[a] > 0.0))
                    throw std::invalid_argument("--cell-size values must be positive metres");
        if(args.origin)
            for(std::size_t a = 0u; a < 3u; ++a)
                if(!std::isfinite((*args.origin)[a]))
                    throw std::invalid_argument("--origin values must be finite metres");
        if(args.globalOrigin)
            for(std::size_t a = 0u; a < 3u; ++a)
                if(!std::isfinite((*args.globalOrigin)[a]))
                    throw std::invalid_argument("--global-origin values must be finite metres");
        if(args.globalGrid)
            for(std::size_t a = 0u; a < 3u; ++a)
                if(std::size_t const n = (*args.globalGrid)[a]; n == 0u || n % 2u != 0u)
                    throw std::invalid_argument("--global-grid values must be even and positive");
        if(args.unitLength && !(*args.unitLength > 0.0))
            throw std::invalid_argument("--picongpu-unit-length must be positive");
        if(args.unitEField && !(*args.unitEField > 0.0))
            throw std::invalid_argument("--picongpu-unit-e must be positive");
        if(args.unitBField && !(*args.unitBField > 0.0))
            throw std::invalid_argument("--picongpu-unit-b must be positive");
        return args;
    }

    void printGrid(picBunch::Grid const& grid)
    {
        std::printf("grid: n=(%zu,%zu,%zu)  spacing=(%.6g,%.6g,%.6g) m\n",
                    grid.n[0], grid.n[1], grid.n[2],
                    grid.spacing[0], grid.spacing[1], grid.spacing[2]);
        std::printf("      origin=(%.6g,%.6g,%.6g) m\n",
                    grid.origin[0], grid.origin[1], grid.origin[2]);
    }

    void printTimings(picBunch::StageTimes const& times)
    {
        std::printf("stage timings (ms):\n");
        std::printf("  %-8s %10.3f\n", "reader", times.readerMs);
        std::printf("  %-8s %10.3f\n", "deposit", times.depositMs);
        std::printf("  %-8s %10.3f\n", "kernel", times.kernelMs);
        std::printf("  %-8s %10.3f\n", "fft", times.fftMs);
        std::printf("  %-8s %10.3f\n", "transpose", times.transposeMs);
        std::printf("  %-8s %10.3f\n", "readout", times.readoutMs);
        std::printf("  %-8s %10.3f\n", "writer", times.writerMs);
        double total = times.readerMs + times.depositMs + times.kernelMs + times.fftMs
            + times.transposeMs + times.readoutMs + times.writerMs;
        std::printf("  %-8s %10.3f\n", "total", total);
    }

    using Clock = std::chrono::steady_clock;

    double elapsedMs(Clock::time_point const& start)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    int runConfigured(Arguments const& args)
    {
        picBunch::Config const config = picBunch::loadConfig(*args.config);
        for(auto const& warning : config.warnings)
            std::fprintf(stderr, "picBunch: warning: %s\n", warning.c_str());

        std::printf(
            "config: species=%s step=%llu precision=%s\n",
            config.picongpu.species.c_str(),
            static_cast<unsigned long long>(config.picongpu.step),
            config.picongpu.float64 ? "float64" : "float32");

        picBunch::BunchSelection selection;
        selection.speciesName = config.picongpu.species;
        picBunch::BunchData const bunchData = picBunch::readBunch(args.input, selection);
        std::printf(
            "read %zu particles (species %s, iteration %llu)\n",
            bunchData.bunch.size(),
            config.picongpu.species.c_str(),
            static_cast<unsigned long long>(bunchData.iteration));

        picBunch::PlacedBunch const placed = picBunch::placeBunch(bunchData.bunch, config);
        std::printf(
            "centroid (charge-weighted, used): (%.9g, %.9g, %.9g) m\n",
            placed.centroid[0], placed.centroid[1], placed.centroid[2]);
        std::printf(
            "bounding-box centre (reported only): (%.9g, %.9g, %.9g) m\n",
            placed.boundingCentre[0], placed.boundingCentre[1], placed.boundingCentre[2]);
        std::printf(
            "target centre: (%.9g, %.9g, %.9g) m; shift: (%.9g, %.9g, %.9g) m\n",
            placed.targetCentre[0], placed.targetCentre[1], placed.targetCentre[2],
            placed.shift[0], placed.shift[1], placed.shift[2]);
        std::printf(
            "particle patch: offset=(%lld,%lld,%lld) cells extent=(%llu,%llu,%llu) cells\n",
            static_cast<long long>(placed.patchOffset[0]),
            static_cast<long long>(placed.patchOffset[1]),
            static_cast<long long>(placed.patchOffset[2]),
            static_cast<unsigned long long>(placed.patchExtent[0]),
            static_cast<unsigned long long>(placed.patchExtent[1]),
            static_cast<unsigned long long>(placed.patchExtent[2]));

        picBunch::ComputeOptions options;
        // The checkpoint writer consumes the full field on rank 0.
        options.fftMode = picBunch::FftMode::Replicated;
        options.tolerance = config.accuracy.tolerance;
        options.fftPadding = args.fftPadding;
        options.cellSize = config.domain.cellSize;
        options.gridSize = config.domain.cells;
        options.gridOrigin = config.domain.origin;
        options.nearShell.mode = (config.accuracy.nearShell == "exact")
            ? picBunch::NearShellOptions::Mode::Exact
            : (config.accuracy.nearShell == "fast")
                ? picBunch::NearShellOptions::Mode::Fast
                : (config.accuracy.nearShell == "fast-gpu")
                    ? picBunch::NearShellOptions::Mode::FastGpu
                    : picBunch::NearShellOptions::Mode::None;
        options.nearShell.fastOrder = config.accuracy.nearShellFastOrder;
        options.nearShell.fastObliqueOrder = config.accuracy.nearShellObliqueOrder;
        options.nearShell.radiusCells = config.accuracy.nearShellRadius;
        options.nearShell.tileNodes = config.accuracy.nearShellTile;
        options.nearShell.exactCellReach = config.accuracy.nearShellExactCells;
        std::printf(
            "field grid: cells=(%zu,%zu,%zu) cell_size=(%.9g,%.9g,%.9g) m "
            "origin=(%.9g,%.9g,%.9g) m\n",
            config.domain.cells[0], config.domain.cells[1], config.domain.cells[2],
            config.domain.cellSize[0], config.domain.cellSize[1], config.domain.cellSize[2],
            config.domain.origin[0], config.domain.origin[1], config.domain.origin[2]);
        picBunch::Field const field = picBunch::computeField(placed.bunch, options);

        picBunch::PicOnGPUWriteOptions restart;
        restart.step = config.picongpu.step;
        restart.directory = *args.destination;
        restart.inputFile = args.input;
        restart.inputIteration = bunchData.iteration;
        restart.species = config.picongpu.species;
        restart.time = static_cast<double>(config.picongpu.step) * config.picongpu.dt;
        restart.timeUnitSI = 1.0;
        restart.dt = config.picongpu.dt;
        restart.unitLength = config.picongpu.unitLength;
        restart.unitEField = config.picongpu.unitEField;
        restart.unitBField = config.picongpu.unitBField;
        restart.unitTime = config.picongpu.unitTime;
        restart.unitSpeed = config.picongpu.unitSpeed;
        restart.unitEnergy = config.picongpu.unitEnergy;
        restart.unitMass = config.picongpu.unitMass;
        restart.unitCharge = config.picongpu.unitCharge;
        restart.float64 = config.picongpu.float64;

        picBunch::GeneratedParticles generated;
        generated.species = config.picongpu.species;
        generated.bunch = placed.bunch;
        generated.domainOrigin = config.domain.origin;
        generated.cellSize = config.domain.cellSize;
        generated.patchOffset = placed.patchOffset;
        generated.patchExtent = placed.patchExtent;
        generated.patches = placed.patches;
        generated.unitMassSpeed = config.picongpu.unitMass * config.picongpu.unitSpeed;
        generated.unitCharge = config.picongpu.unitCharge;
        generated.unitMass = config.picongpu.unitMass;
        generated.fallbackMass = config.picongpu.unitMass;
        restart.generatedParticles = generated;

        picBunch::writePicOnGPUCheckpoint(field, restart);
        std::printf(
            "wrote checkpoint step %llu to %s (%s particle records, %s fields)\n",
            static_cast<unsigned long long>(config.picongpu.step),
            args.destination->c_str(),
            config.picongpu.float64 ? "float64" : "float32",
            config.picongpu.float64 ? "float64" : "float32");
        return 0;
    }

    int runChunked(Arguments const& args)
    {
        if(!args.output)
            throw std::runtime_error("chunked mode requires --output");
        if(args.globalGrid)
            throw std::runtime_error(
                "--global-grid is not combined with --max-particle-chunk-gb yet");
        if(!args.grid || !args.origin || !args.cellSize)
            throw std::runtime_error(
                "chunked mode requires explicit --grid, --origin and --cell-size so "
                "every chunk uses the same field grid");
        if(args.velocityBins != 1u)
            throw std::runtime_error(
                "chunked mode currently requires --velocity-bins 1; a multi-velocity "
                "chunked run needs a global binning pass, which is not implemented yet");

        bool const root = picBunch::mpiRank() == 0;
        picBunch::BunchSelection selection;
        selection.iteration = args.inputIteration;
        if(args.species)
        {
            if(isInteger(*args.species))
                selection.speciesIndex = static_cast<std::size_t>(
                    parseIteration(*args.species, "--species index"));
            else
                selection.speciesName = *args.species;
        }

        picBunch::StageTimes times;
        std::uint64_t totalCount = 0;
        if(root)
            totalCount = picBunch::readParticleCount(args.input, selection);
        double totalMeta = static_cast<double>(totalCount);
        picBunch::mpiBroadcast(&totalMeta, 1, 0);
        totalCount = static_cast<std::uint64_t>(totalMeta);
        if(totalCount == 0)
            throw std::runtime_error("chunked mode: the selected species is empty");

        constexpr std::size_t bytesPerParticle = 9u * sizeof(double);
        std::size_t chunkSize = static_cast<std::size_t>(totalCount);
        if(args.maxParticleChunkGb > 0.0)
        {
            double const budget = args.maxParticleChunkGb * 1.0e9;
            std::size_t const byBudget
                = static_cast<std::size_t>(budget / static_cast<double>(bytesPerParticle));
            chunkSize = std::max<std::size_t>(1u, std::min<std::size_t>(chunkSize, byBudget));
        }
        std::size_t const chunks
            = (chunkSize == 0u) ? 0u : (static_cast<std::size_t>(totalCount) + chunkSize - 1u) / chunkSize;
        if(root)
            std::printf(
                "chunked reader: %llu particles, %zu chunk(s), at most %zu particles "
                "(%.3f GB) per chunk\n",
                static_cast<unsigned long long>(totalCount),
                chunks,
                chunkSize,
                static_cast<double>(chunkSize * bytesPerParticle) / 1.0e9);

        picBunch::MpiParallelContext parallel;
        picBunch::ComputeOptions options;
        options.tolerance = args.tolerance;
        options.fftPadding = args.fftPadding;
        options.deterministicDeposit = args.deterministicDeposit;
        options.shape = (args.shape == "cic") ? picBunch::Shape::CIC
            : (args.shape == "pqs") ? picBunch::Shape::PQS
                                    : picBunch::Shape::TSC;
        options.cellSize = *args.cellSize;
        options.gridSize = *args.grid;
        options.gridOrigin = *args.origin;
        options.velocityBinning.mode = picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = 1u;
        options.velocityBinning.modelConstant = args.binModelConstant;
        options.velocityBinning.maxBins = args.maxVelocityBins;
        options.nearShell.mode = (args.nearShell == "exact")
            ? picBunch::NearShellOptions::Mode::Exact
            : (args.nearShell == "fast")
                ? picBunch::NearShellOptions::Mode::Fast
                : (args.nearShell == "fast-gpu")
                    ? picBunch::NearShellOptions::Mode::FastGpu
                    : picBunch::NearShellOptions::Mode::None;
        options.nearShell.fastOrder = args.nearShellFastOrder;
        options.nearShell.fastObliqueOrder = (args.nearShellObliqueOrder == "auto")
            ? 0
            : std::stoi(args.nearShellObliqueOrder);
        options.nearShell.radiusCells = args.nearShellRadius;
        options.nearShell.tileNodes = args.nearShellTile;
        options.nearShell.exactCellReach = args.nearShellExactCells;
        options.fftMode = (args.fftMode == "replicated")
            ? picBunch::FftMode::Replicated
            : (args.fftMode == "distributed") ? picBunch::FftMode::Distributed
                                              : picBunch::FftMode::Auto;
        options.parallel = &parallel;

        // Global representative for the single velocity class.  The chunk
        // fields are linear only if every chunk uses the same kernel, so the
        // representative must be derived from the whole species, not per chunk.
        Vec3 velocitySum{0.0, 0.0, 0.0};
        double velocityWeight = 0.0;
        for(std::size_t offset = 0; offset < static_cast<std::size_t>(totalCount);
            offset += chunkSize)
        {
            std::size_t const count
                = std::min<std::size_t>(chunkSize, static_cast<std::size_t>(totalCount) - offset);
            picBunch::BunchData chunk;
            if(root)
                chunk = picBunch::readBunchRange(args.input, selection, offset, count);
            picBunch::mpiBroadcastBunch(chunk.bunch, 0);
            for(std::size_t p = 0; p < chunk.bunch.size(); ++p)
            {
                double const w = std::abs(chunk.bunch.weight[p] * chunk.bunch.charge[p]);
                velocitySum[0] += w * chunk.bunch.ux[p];
                velocitySum[1] += w * chunk.bunch.uy[p];
                velocitySum[2] += w * chunk.bunch.uz[p];
                velocityWeight += w;
            }
        }
        if(velocityWeight > 0.0)
            for(int a = 0; a < 3; ++a)
                velocitySum[a] /= velocityWeight;
        picBunch::VelocityBinning fixedBinning;
        fixedBinning.derived = true;
        fixedBinning.count = 1u;
        fixedBinning.perAxis = {1u, 1u, 1u};
        fixedBinning.representativeU.resize(1u);
        fixedBinning.representativeU[0] = velocitySum;
        options.fixedBinning = fixedBinning;

        picBunch::Field totalField;
        double metaTime = 0.0;
        double metaTimeUnitSI = 1.0;
        std::uint64_t metaIteration = 0;
        bool haveMeta = false;
        for(std::size_t offset = 0; offset < static_cast<std::size_t>(totalCount);
            offset += chunkSize)
        {
            std::size_t const count
                = std::min<std::size_t>(chunkSize, static_cast<std::size_t>(totalCount) - offset);
            picBunch::BunchData chunk;
            if(root)
            {
                auto const t0 = Clock::now();
                chunk = picBunch::readBunchRange(args.input, selection, offset, count);
                times.readerMs += elapsedMs(t0);
            }
            picBunch::mpiBroadcastBunch(chunk.bunch, 0);
            double chunkMeta[3] = {
                chunk.time,
                chunk.timeUnitSI,
                static_cast<double>(chunk.iteration)};
            if(root)
            {
                chunkMeta[0] = chunk.time;
                chunkMeta[1] = chunk.timeUnitSI;
                chunkMeta[2] = static_cast<double>(chunk.iteration);
            }
            picBunch::mpiBroadcast(chunkMeta, 3, 0);
            if(!root)
            {
                chunk.time = chunkMeta[0];
                chunk.timeUnitSI = chunkMeta[1];
                chunk.iteration = static_cast<std::uint64_t>(chunkMeta[2]);
            }
            if(!haveMeta)
            {
                metaTime = chunk.time;
                metaTimeUnitSI = chunk.timeUnitSI;
                metaIteration = chunk.iteration;
                haveMeta = true;
            }

            picBunch::StageTimes chunkTimes;
            picBunch::Field const part
                = picBunch::computeField(chunk.bunch, options, &chunkTimes);
            times.depositMs += chunkTimes.depositMs;
            times.kernelMs += chunkTimes.kernelMs;
            times.fftMs += chunkTimes.fftMs;
            times.readoutMs += chunkTimes.readoutMs;
            times.transposeMs += chunkTimes.transposeMs;
            if(totalField.data[0].empty())
            {
                totalField = part;
            }
            else
            {
                for(int c = 0; c < 6; ++c)
                {
                    if(totalField.data[c].size() != part.data[c].size())
                        throw std::runtime_error(
                            "chunked mode: field sizes differ between chunks");
                    for(std::size_t l = 0; l < part.data[c].size(); ++l)
                        totalField.data[c][l] += part.data[c][l];
                }
            }
            if(root)
                std::printf(
                    "chunk %zu: %zu particles\n",
                    offset / chunkSize + 1u,
                    chunk.bunch.size());
        }
        if(totalField.data[0].empty())
            throw std::runtime_error("chunked mode: no particle chunk produced a field");

        std::array<double, 2> peaks{0.0, 0.0};
        for(int c = 0; c < 6; ++c)
            for(double value : totalField.data[c])
                peaks[c < 3 ? 0 : 1] = std::max(peaks[c < 3 ? 0 : 1], std::abs(value));
        if(options.parallel && options.parallel->distributed())
            options.parallel->allreduceMax(peaks.data(), peaks.size());
        if(root)
        {
            std::printf(
                "field: max|E|=%g V/m  max|B|=%g T\n", peaks[0], peaks[1]);
        }

        picBunch::FieldWriteOptions writeOptions;
        writeOptions.iteration = args.outputIteration;
        writeOptions.time = metaTime;
        writeOptions.timeUnitSI = metaTimeUnitSI;
        auto const writerStart = Clock::now();
        if(totalField.zSlab)
            picBunch::writeFieldToDomain(
                totalField, *args.output, writeOptions, totalField.grid.n, totalField.grid.origin);
        else if(root)
            picBunch::writeField(totalField, *args.output, writeOptions);
        times.writerMs = elapsedMs(writerStart);
        if(root)
        {
            std::printf(
                "wrote %s (iteration %llu)\n",
                args.output->c_str(),
                static_cast<unsigned long long>(args.outputIteration));
            printTimings(times);
        }
        return 0;
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
        std::fprintf(stderr, "picBunch: %s\n\n", error.what());
        usage(stderr);
        return 2;
    }

    if(args.config)
    {
        try
        {
            return runConfigured(args);
        }
        catch(std::exception const& error)
        {
            std::fprintf(stderr, "picBunch: error: %s\n", error.what());
            return 1;
        }
    }

    picBunch::MpiGuard mpiRuntime(argc, argv);
    if(args.maxParticleChunkGb > 0.0)
    {
        try
        {
            return runChunked(args);
        }
        catch(std::exception const& error)
        {
            std::fprintf(stderr, "picBunch: error: %s\n", error.what());
            return 1;
        }
    }
    picBunch::StageTimes times;
    try
    {
        picBunch::BunchData bunchData;
        picBunch::Field field;
        std::optional<picBunch::FarTail> farTail;
        bool const root = picBunch::mpiRank() == 0;
        Vec3 globalOriginResolved{0.0, 0.0, 0.0};
        picBunch::MpiParallelContext parallel;
        picBunch::ComputeOptions options;

        std::string rootSetupError;
        if(root)
        {
        try
        {
        // ---------------------------------------------------------------- reader
        auto const readerStart = Clock::now();
        picBunch::BunchSelection selection;
        selection.iteration = args.inputIteration;
        if(args.species)
        {
            if(isInteger(*args.species))
                selection.speciesIndex = static_cast<std::size_t>(
                    parseIteration(*args.species, "--species index"));
            else
                selection.speciesName = *args.species;
        }
        bunchData = picBunch::readBunch(args.input, selection);
        times.readerMs = elapsedMs(readerStart);
        std::printf(
            "read %zu particles (iteration %llu, t=%g s)\n",
            bunchData.bunch.size(),
            static_cast<unsigned long long>(bunchData.iteration),
            bunchData.time * bunchData.timeUnitSI);

        // Geometry: an explicit --cell-size always wins over the input mesh.
        // When neither is available the run fails; nothing is assumed.
        options.tolerance = args.tolerance;
        options.fftPadding = args.fftPadding;
        options.deterministicDeposit = args.deterministicDeposit;
        options.shape = (args.shape == "cic") ? picBunch::Shape::CIC
            : (args.shape == "pqs") ? picBunch::Shape::PQS
                                    : picBunch::Shape::TSC;
        options.nearShell.mode = (args.nearShell == "exact")
            ? picBunch::NearShellOptions::Mode::Exact
            : (args.nearShell == "fast")
                ? picBunch::NearShellOptions::Mode::Fast
                : (args.nearShell == "fast-gpu")
                    ? picBunch::NearShellOptions::Mode::FastGpu
                    : picBunch::NearShellOptions::Mode::None;
        options.nearShell.fastOrder = args.nearShellFastOrder;
        options.nearShell.fastObliqueOrder = (args.nearShellObliqueOrder == "auto")
            ? 0
            : std::stoi(args.nearShellObliqueOrder);
        options.nearShell.radiusCells = args.nearShellRadius;
        options.nearShell.tileNodes = args.nearShellTile;
        options.nearShell.exactCellReach = args.nearShellExactCells;
        options.velocityBinning.mode
            = (args.velocityBins == 0u) ? picBunch::VelocityBinningOptions::Mode::Auto
                                        : picBunch::VelocityBinningOptions::Mode::Explicit;
        options.velocityBinning.explicitBins = std::max<std::size_t>(1u, args.velocityBins);
        options.velocityBinning.modelConstant = args.binModelConstant;
        options.velocityBinning.maxBins = args.maxVelocityBins;
        options.velocityMoments.mode
            = (args.velocityMoments == "auto") ? picBunch::VelocityMomentsOptions::Mode::Auto
            : (args.velocityMoments == "2" || args.velocityMoments == "3")
                ? picBunch::VelocityMomentsOptions::Mode::Explicit
                : picBunch::VelocityMomentsOptions::Mode::Off;
        if(args.velocityMoments == "2" || args.velocityMoments == "3")
            options.velocityMoments.order = std::stoul(args.velocityMoments);
        options.velocityMoments.allowUnconverged = args.velocityMomentsAllowUnconverged;
        std::optional<Vec3> meshCellSize;
        std::optional<Vec3> meshGridOrigin;
        std::optional<Idx3> meshGridExtent;
        try
        {
            auto const geometry
                = picBunch::readMeshGeometry(args.input, "E", 0, bunchData.iteration);
            // readMeshGeometry already returns physical SI spacing and offset
            // (raw gridSpacing/gridGlobalOffset times gridUnitSI) and the shape
            // mapped to x,y,z.  Applying gridUnitSI again here is the v0.1
            // regression; the reader owns the conversion and these helpers only
            // validate the result.
            meshCellSize = picBunch::physicalCellSize(geometry);
            meshGridOrigin = picBunch::physicalGridOrigin(geometry);
            meshGridExtent = picBunch::physicalGridExtent(geometry);
            if(meshCellSize)
                std::printf(
                    "cellSize from input mesh E: gridSpacing=(%.6g,%.6g,%.6g) x "
                    "gridUnitSI=(%.6g,%.6g,%.6g) -> (%.6g,%.6g,%.6g) m\n",
                    geometry.gridSpacing[0] / geometry.gridUnitSI[0],
                    geometry.gridSpacing[1] / geometry.gridUnitSI[1],
                    geometry.gridSpacing[2] / geometry.gridUnitSI[2],
                    geometry.gridUnitSI[0], geometry.gridUnitSI[1], geometry.gridUnitSI[2],
                    (*meshCellSize)[0], (*meshCellSize)[1], (*meshCellSize)[2]);
            else
                std::printf("cellSize: input mesh geometry is unusable\n");
            if(meshGridOrigin)
                std::printf(
                    "origin: input mesh gridGlobalOffset x gridUnitSI = "
                    "(%.6g,%.6g,%.6g) m\n",
                    (*meshGridOrigin)[0], (*meshGridOrigin)[1], (*meshGridOrigin)[2]);
            else
                std::printf("origin: input mesh gridGlobalOffset is unusable\n");
            if(meshGridExtent)
                std::printf(
                    "grid: input mesh extent (%zu,%zu,%zu) nodes\n",
                    (*meshGridExtent)[0], (*meshGridExtent)[1], (*meshGridExtent)[2]);
            else
                std::printf("grid: input mesh extent is unusable\n");
        }
        catch(picBunch::MeshUnavailable const&)
        {
            std::printf("cellSize: input has no mesh\n");
        }
        options.cellSize = picBunch::resolveCellSize(args.cellSize, meshCellSize);
        if(args.cellSize)
            std::printf(
                "cellSize: --cell-size override (%.6g,%.6g,%.6g) m\n",
                options.cellSize[0], options.cellSize[1], options.cellSize[2]);
        // Anisotropic cells are valid, but the near-field error scales with the
        // distance measured in local cell units.  Warn when the spacing ratio is
        // large enough that the shortest axis, not the requested "d cells",
        // controls the near-source accuracy.  This changes no computation.
        {
            double shortest = options.cellSize[0];
            double largest = options.cellSize[0];
            for(std::size_t a = 1u; a < 3u; ++a)
            {
                shortest = std::min(shortest, options.cellSize[a]);
                largest = std::max(largest, options.cellSize[a]);
            }
            double const ratio = largest / shortest;
            if(ratio >= 2.0)
                std::fprintf(
                    stderr,
                    "picBunch: warning: anisotropic cell spacing: largest %.6g m, "
                    "shortest %.6g m (ratio %.3g); near-field accuracy is set by the "
                    "shortest axis (%.6g m)\n",
                    largest, shortest, ratio, shortest);
        }
        // The mesh extent is the default computation grid; --grid is an explicit
        // override.  A --global-grid must not enlarge the computation, so the
        // mesh extent is ignored for the compute box in that case.
        if(args.fftMode == "replicated")
            options.fftMode = picBunch::FftMode::Replicated;
        else if(args.fftMode == "distributed")
            options.fftMode = picBunch::FftMode::Distributed;
        else
            options.fftMode = picBunch::FftMode::Auto;
        if(args.restartFormat == "picongpu")
        {
            // The checkpoint writer needs all six full components on rank 0;
            // a slab-decomposed field cannot feed it without a gather.
            options.fftMode = picBunch::FftMode::Replicated;
            if(root && args.fftMode != "replicated")
                std::printf(
                    "fft mode: replicated (PIConGPU checkpoint needs the full field)\n");
        }
        options.meshGridSize = args.globalGrid ? std::nullopt : meshGridExtent;
        if(args.grid)
        {
            options.gridSize = args.grid;
            std::printf(
                "grid: --grid override (%zu,%zu,%zu) nodes\n",
                (*args.grid)[0], (*args.grid)[1], (*args.grid)[2]);
        }
        // The mesh origin is the default; --origin is an explicit override.  With
        // neither, makeGrid keeps the bunch-bounding-box-minus-margin origin.
        // With a --global-grid the computation must stay on the bunch box, so the
        // mesh origin is not allowed to displace the derived box unless the user
        // asked for an explicit computation origin.
        options.meshGridOrigin
            = (args.globalGrid && !args.origin) ? std::nullopt : meshGridOrigin;
        options.gridOrigin = args.origin;
        if(args.origin)
            std::printf(
                "origin: --origin override (%.6g,%.6g,%.6g) m\n",
                (*args.origin)[0], (*args.origin)[1], (*args.origin)[2]);
        else if(meshGridOrigin)
            std::printf(
                "origin: input mesh default (%.6g,%.6g,%.6g) m\n",
                (*meshGridOrigin)[0], (*meshGridOrigin)[1], (*meshGridOrigin)[2]);

        if(args.globalGrid)
        {
            if(args.globalOrigin)
                globalOriginResolved = *args.globalOrigin;
            else if(meshGridOrigin)
                globalOriginResolved = *meshGridOrigin;
            else
                globalOriginResolved = {0.0, 0.0, 0.0};
            options.globalGridSize = args.globalGrid;
            options.globalGridOrigin = globalOriginResolved;
            std::printf(
                "global domain: n=(%zu,%zu,%zu) origin=(%.6g,%.6g,%.6g) m "
                "(chunked write, computation stays on the bunch box)\n",
                (*args.globalGrid)[0], (*args.globalGrid)[1], (*args.globalGrid)[2],
                globalOriginResolved[0], globalOriginResolved[1], globalOriginResolved[2]);
        }

        } // root: reader and geometry resolution
        catch(std::exception const& error)
        {
            rootSetupError = error.what();
        }
        catch(...)
        {
            rootSetupError = "unknown error during root setup";
        }
        }
        // Publish the root setup outcome to every rank before the first data
        // collective.  Without this a root-only reader/config/geometry failure
        // leaves the other ranks waiting in the broadcast.
        {
            std::string setupMessage = rootSetupError;
            int const setupStatus = picBunch::mpiBroadcastRootSetupStatus(setupMessage, 0);
            if(setupStatus != 0)
                throw std::runtime_error(
                    setupMessage.empty() ? "root setup failed" : setupMessage);
        }

        // The bunch is replicated, not redistributed: the near shell and the
        // component-parallel mesh both need every source on every rank.  Only
        // the small resolved option set travels with it.
        picBunch::mpiBroadcastBunch(bunchData.bunch, 0);
        {
            double meta[3] = {
                bunchData.time,
                bunchData.timeUnitSI,
                static_cast<double>(bunchData.iteration)};
            picBunch::mpiBroadcast(meta, 3, 0);
            if(!root)
            {
                bunchData.time = meta[0];
                bunchData.timeUnitSI = meta[1];
                bunchData.iteration = static_cast<std::uint64_t>(meta[2]);
            }
        }
        picBunch::mpiBroadcastComputeOptions(options, 0);
        options.parallel = &parallel;

        // --------------------------------------------------------------- compute
        auto const computeStart = Clock::now();
        if(root)
        {
            std::printf(
                "near shell: mode=%s R=%.3g PIC cells tile=%zu nodes\n",
                args.nearShell.c_str(),
                picBunch::resolveNearShellRadius(options, options.nearShell),
                options.nearShell.tileNodes);
            // The near shell is an opt-in correction with a regime-dependent
            // accuracy envelope.  State it instead of letting --tolerance imply
            // an in-support guarantee it does not provide.
            bool moving = false;
            for(std::size_t p = 0; p < bunchData.bunch.size() && !moving; ++p)
                moving = bunchData.bunch.ux[p] != 0.0 || bunchData.bunch.uy[p] != 0.0
                    || bunchData.bunch.uz[p] != 0.0;
            if(options.nearShell.mode == picBunch::NearShellOptions::Mode::Exact && moving)
                std::fprintf(
                    stderr,
                    "picBunch: warning: --near-shell exact integrates moving in-support\n"
                    "  points in Lorentz-contracted coordinates.  Tested agreement is about\n"
                    "  1e-4 at cells (1,0,0) and (0,0.5,0) and about 2%% at the closest\n"
                    "  in-support node for gamma 10; it is not a blanket 1e-3 path.  Validate\n"
                    "  production in-support points with tools/velocity-bins/indfield.py.\n");
            if((options.nearShell.mode == picBunch::NearShellOptions::Mode::Fast
                || options.nearShell.mode == picBunch::NearShellOptions::Mode::FastGpu)
               && options.tolerance < 1.0e-3)
                std::fprintf(
                    stderr,
                    "picBunch: warning: --near-shell fast order %d has a fixed in-support\n"
                    "  envelope (about 8e-4 stationary, 1.7e-4 moving axial, 8e-2 moving\n"
                    "  oblique); --tolerance %g is not met inside the TSC support.\n",
                    options.nearShell.fastOrder,
                    options.tolerance);
        }
        picBunch::VelocityBinning binning;
        picBunch::VelocityMomentsReport velocityMomentsReport;
        if(options.nearShell.mode != picBunch::NearShellOptions::Mode::None)
        {
            std::size_t exactPairs = 0;
            options.nearShell.exactPairs = &exactPairs;
            field = picBunch::computeField(
                bunchData.bunch, options, &times, &binning, &velocityMomentsReport);
            if(root)
                std::printf(
                    "near shell: %zu exact (source,node,component) evaluations\n",
                    exactPairs);
        }
        else
            field = picBunch::computeField(
                bunchData.bunch, options, &times, &binning, &velocityMomentsReport);
        if(root && options.velocityMoments.mode != picBunch::VelocityMomentsOptions::Mode::Off)
        {
            if(velocityMomentsReport.used)
                std::printf(
                    "velocity moments: used order %d, u0=(%.6g, %.6g, %.6g) m/c, spread %.3g%s, "
                    "probe %.3e%s\n",
                    velocityMomentsReport.order,
                    velocityMomentsReport.u0[0],
                    velocityMomentsReport.u0[1],
                    velocityMomentsReport.u0[2],
                    velocityMomentsReport.maxSpread,
                    options.velocityMoments.mode == picBunch::VelocityMomentsOptions::Mode::Auto
                        ? " (auto)"
                        : "",
                    velocityMomentsReport.probe,
                    velocityMomentsReport.allowUnconverged ? " [UNCONVERGED: forced]" : "");
            else
                std::printf(
                    "velocity moments: fell back to the binned path (%s)\n",
                    velocityMomentsReport.nonFinite
                        ? "non-finite moment field"
                        : velocityMomentsReport.feasible
                            ? "order probe above the 0.05 budget"
                            : "infeasible (memory or size)");
        }
        if(binning.derived)
            std::printf(
                "velocity bins: %zu box(es) (%zux%zux%zu), model estimate %.3e "
                "(constant %.3e, tolerance %.3e)%s\n",
                binning.count,
                binning.perAxis[0],
                binning.perAxis[1],
                binning.perAxis[2],
                binning.modelError,
                binning.modelConstant,
                args.tolerance,
                binning.multiVelocity ? "" : ", single velocity");

        auto fieldPeaks = [&](picBunch::Field const& value) {
            std::array<double, 2> peaks{0.0, 0.0};
            for(int c = 0; c < 6; ++c)
                for(double entry : value.data[c])
                    peaks[c < 3 ? 0 : 1] = std::max(peaks[c < 3 ? 0 : 1], std::abs(entry));
            // A distributed field holds one z-slab per rank; the physical peak
            // is the maximum over all slabs.
            if(options.parallel && options.parallel->distributed())
                options.parallel->allreduceMax(peaks.data(), peaks.size());
            return peaks;
        };
        auto peaks = fieldPeaks(field);
        double maxE = peaks[0];
        double maxB = peaks[1];

        // The chunked writer fills the global domain outside the compute box
        // with exact zero.  That is only physical at the stated tolerance once
        // the compute box extends to the tolerance-derived tail radius.  Expand
        // the box here (second pass); if the global domain ends before the
        // radius, makeGrid clips the box to the domain, so no zero is written
        // between the bunch and the domain edge.
        if(args.globalGrid)
        {
            // The same E-tail bound also bounds B: |B_tail| <= |E_tail|/c.
            // Requiring E_tail <= tol * min(E_peak, c B_peak) therefore keeps
            // both per-record relative tails at or below tolerance.  A zero B
            // peak means an electrostatic field and is ignored.
            double peak = maxE;
            if(maxB > 0.0)
                peak = std::min(peak, picBunch::kSpeedOfLight * maxB);
            picBunch::FieldTailMargin const margin = picBunch::deriveFieldTailMargin(
                bunchData.bunch, peak, args.tolerance, options.cellSize);
            std::printf(
                "field tail: |Q|=%g C, |p|=%g C m, R=%g m, gamma_max=%g, target=%g V/m, "
                "d=%g m, bound=%g V/m, margin=(%.6g,%.6g,%.6g) cells\n",
                std::abs(margin.totalCharge), margin.dipoleMagnitude, margin.sourceRadius,
                margin.maxGamma, args.tolerance * peak, margin.distance, margin.bound,
                margin.marginCells[0], margin.marginCells[1], margin.marginCells[2]);
            // The multipole tail replaces the former zero fill outside the
            // exact compute box.  Its measured validity radius is a multiple
            // of the bunch support radius; the exact box must reach it (plus
            // two cells for the global-node snap) before the tail is used.
            picBunch::FarTail const tail = picBunch::deriveFarTail(
                bunchData.bunch,
                6.0,
                options.cellSize,
                args.multipoleGroupCells);
            auto const bounds = bunchData.bunch.bounds();
            Vec3 tailMargin{};
            for(int a = 0; a < 3; ++a)
            {
                double const halfExtent = 0.5 * (bounds[1][a] - bounds[0][a]);
                double const cells = (tail.switchDistance - halfExtent) / options.cellSize[a];
                tailMargin[a] = std::max(0.0, std::ceil(cells) + 2.0);
            }
            std::printf(
                "far tail: R=%g m, switch=%g m (%g cells), quadrupole norm=%g C m^2, "
                "tail margin=(%.6g,%.6g,%.6g) cells\n",
                tail.supportRadius, tail.switchDistance,
                tail.switchDistance / options.cellSize[0], margin.quadrupoleNorm,
                tailMargin[0], tailMargin[1], tailMargin[2]);

            if(args.allowFieldTruncation)
                std::fprintf(
                    stderr,
                    "picBunch: warning: --allow-field-truncation: keeping the "
                    "unexpanded compute box and writing exact zero outside it; the "
                    "neglected field there is not bounded by --tolerance %g\n",
                    args.tolerance);
            else
            {
                auto const computeWithMargin = [&](Vec3 const& useMargin) {
                    picBunch::ComputeOptions expanded = options;
                    expanded.marginCellsPerAxis = useMargin;
                    // The margin, not the previous explicit extent/origin, now
                    // defines the compute box.  The global clip in makeGrid keeps
                    // it inside the written domain.
                    expanded.gridSize.reset();
                    expanded.gridOrigin.reset();
                    expanded.meshGridSize.reset();
                    expanded.meshGridOrigin.reset();
                    field = picBunch::computeField(bunchData.bunch, expanded, &times);
                    peaks = fieldPeaks(field);
                    maxE = peaks[0];
                    maxB = peaks[1];
                };
                Vec3 preferred = margin.marginCells;
                bool needsExpansion = false;
                for(int a = 0; a < 3; ++a)
                {
                    preferred[a] = std::max(preferred[a], tailMargin[a]);
                    if(preferred[a] > options.marginCells)
                        needsExpansion = true;
                }
                if(needsExpansion)
                {
                    try
                    {
                        computeWithMargin(preferred);
                    }
                    catch(std::runtime_error const& error)
                    {
                        // The quadrupole margin can be physically justified yet
                        // exceed the padded-node cap.  The far tail stays valid
                        // at its measured switch radius, so fall back to that
                        // smaller exact box instead of failing.
                        std::string const message = error.what();
                        bool const infeasible
                            = message.find("infeasible") != std::string::npos;
                        bool smaller = false;
                        for(int a = 0; a < 3; ++a)
                            if(tailMargin[a] < preferred[a])
                                smaller = true;
                        if(!infeasible || !smaller)
                            throw std::runtime_error(
                                std::string("tolerance-derived field margin cannot be computed: ")
                                + message
                                + "; enlarge the compute box or pass --allow-field-truncation"
                                  " to acknowledge the truncation");
                        std::printf(
                            "field tail: quadrupole margin exceeds the padded-node cap; "
                            "using the measured far-tail switch at (%.6g,%.6g,%.6g) cells\n",
                            tailMargin[0], tailMargin[1], tailMargin[2]);
                        computeWithMargin(tailMargin);
                    }
                }
                farTail = tail;
            }
        }

        double const computeWallMs = elapsedMs(computeStart);
        if(root)
        {
            printGrid(field.grid);
            std::printf("field: max|E|=%g V/m  max|B|=%g T\n", maxE, maxB);
            std::printf("compute wall (all stages, rank 0): %.3f ms\n", computeWallMs);
        }

        // A distributed run leaves only the rank's output z-slab on each rank;
        // a replicated run leaves the full bunch-box field on every rank.  The
        // writer consumes either form without a full-field broadcast.
        double scalarMeta[6] = {
            bunchData.time,
            bunchData.timeUnitSI,
            static_cast<double>(bunchData.iteration),
            globalOriginResolved[0],
            globalOriginResolved[1],
            globalOriginResolved[2]};
        picBunch::mpiBroadcast(scalarMeta, 6, 0);
        if(!root)
        {
            bunchData.time = scalarMeta[0];
            bunchData.timeUnitSI = scalarMeta[1];
            bunchData.iteration = static_cast<std::uint64_t>(scalarMeta[2]);
            globalOriginResolved = {scalarMeta[3], scalarMeta[4], scalarMeta[5]};
        }
        // Every rank fills its own global chunk, so the far-tail moments must be
        // identical everywhere.  Broadcast the small fixed-size description.
        {
            auto packTail = [](picBunch::FarTail const& tail, double* data) {
                data[0] = 1.0;
                for(int a = 0; a < 3; ++a)
                {
                    data[1 + a] = tail.centre[a];
                    data[5 + a] = tail.dipole[a];
                    data[17 + a] = tail.meanVelocity[a];
                }
                data[4] = tail.charge;
                for(int i = 0; i < 9; ++i)
                    data[8 + i] = tail.quadrupole[i];
                data[20] = tail.supportRadius;
                data[21] = tail.switchDistance;
                data[22] = tail.gammaMax;
            };
            auto unpackTail = [](double const* data, picBunch::FarTail& tail) {
                for(int a = 0; a < 3; ++a)
                {
                    tail.centre[a] = data[1 + a];
                    tail.dipole[a] = data[5 + a];
                    tail.meanVelocity[a] = data[17 + a];
                }
                tail.charge = data[4];
                for(int i = 0; i < 9; ++i)
                    tail.quadrupole[i] = data[8 + i];
                tail.supportRadius = data[20];
                tail.switchDistance = data[21];
                tail.gammaMax = data[22];
            };
            double tailData[23] = {};
            tailData[0] = farTail ? 1.0 : 0.0;
            if(farTail)
                packTail(*farTail, tailData);
            picBunch::mpiBroadcast(tailData, 23, 0);
            if(!root && tailData[0] != 0.0)
            {
                farTail.emplace();
                unpackTail(tailData, *farTail);
            }
            // Spatial group multipoles must be identical on every rank.
            double groupCount = farTail ? static_cast<double>(farTail->groups.size()) : 0.0;
            picBunch::mpiBroadcast(&groupCount, 1, 0);
            std::size_t const groups = static_cast<std::size_t>(groupCount);
            std::vector<double> groupData(groups * 23u);
            if(root && farTail)
                for(std::size_t g = 0; g < groups; ++g)
                    packTail(farTail->groups[g], groupData.data() + g * 23u);
            if(!groupData.empty())
                picBunch::mpiBroadcast(groupData.data(), groupData.size(), 0);
            if(!root && farTail)
            {
                farTail->groups.resize(groups);
                for(std::size_t g = 0; g < groups; ++g)
                    unpackTail(groupData.data() + g * 23u, farTail->groups[g]);
            }
        }

        // ---------------------------------------------------------------- writer
        if(args.restartFormat == "picongpu" && root)
        {
            std::vector<char const*> missing;
            if(!args.unitLength)
                missing.push_back("--picongpu-unit-length");
            if(!args.unitEField)
                missing.push_back("--picongpu-unit-e");
            if(!args.unitBField)
                missing.push_back("--picongpu-unit-b");
            if(!missing.empty())
            {
                std::string message = "--restart-format picongpu requires";
                for(auto const* name : missing)
                    message += std::string(" ") + name;
                throw std::invalid_argument(message + "; no unit is inferred");
            }
            if(args.picongpuDt < 0.0 || !std::isfinite(args.picongpuDt))
                throw std::invalid_argument("--picongpu-dt must be finite and non-negative");

            picBunch::PicOnGPUWriteOptions restart;
            restart.step = args.picongpuStep.value_or(args.outputIteration);
            restart.directory = args.outputDir;
            restart.inputFile = args.input;
            restart.inputIteration = bunchData.iteration;
            if(args.picongpuSpecies)
                restart.species = *args.picongpuSpecies;
            else if(args.species && !isInteger(*args.species))
                restart.species = *args.species;
            restart.time = bunchData.time;
            restart.timeUnitSI = bunchData.timeUnitSI;
            restart.dt = args.picongpuDt;
            restart.unitLength = *args.unitLength;
            restart.unitEField = *args.unitEField;
            restart.unitBField = *args.unitBField;
            // The writer derives nextId from the particles actually copied, so
            // the id range covers every species.  Do not override it with the
            // size of the bunch species read for the field calculation.

            auto const writerStart = Clock::now();
            picBunch::writePicOnGPUCheckpoint(field, restart);
            times.writerMs = elapsedMs(writerStart);
            std::printf(
                "wrote PIConGPU restart checkpoint %s/%s_%06llu.h5 (step %llu)\n",
                args.outputDir.c_str(),
                restart.prefix.c_str(),
                static_cast<unsigned long long>(restart.step),
                static_cast<unsigned long long>(restart.step));
        }
        else if(args.output)
        {
            picBunch::FieldWriteOptions writeOptions;
            writeOptions.iteration = args.outputIteration;
            writeOptions.time = bunchData.time;
            writeOptions.timeUnitSI = bunchData.timeUnitSI;
            auto const writerStart = Clock::now();
            if(args.globalGrid)
                picBunch::writeFieldToDomain(
                    field,
                    *args.output,
                    writeOptions,
                    *args.globalGrid,
                    globalOriginResolved,
                    farTail ? &*farTail : nullptr);
            else if(field.zSlab)
                // Distributed output: every rank holds its own z-slab.  The
                // chunked writer writes it collectively without materialising
                // the full domain.
                picBunch::writeFieldToDomain(
                    field,
                    *args.output,
                    writeOptions,
                    field.grid.n,
                    field.grid.origin);
            else if(picBunch::mpiSize() > 1)
            {
                // Replicated field: every rank holds the full array, so each
                // rank can write its own z-slab in parallel.  Copy only the
                // owned slab into a view and use the collective chunked writer.
                std::size_t const nx = field.grid.n[0];
                std::size_t const ny = field.grid.n[1];
                std::size_t const nz = field.grid.n[2];
                std::size_t const plane = nx * ny;
                int const rank = picBunch::mpiRank();
                int const size = picBunch::mpiSize();
                std::size_t const z0
                    = (static_cast<std::size_t>(rank) * nz) / static_cast<std::size_t>(size);
                std::size_t const z1
                    = ((static_cast<std::size_t>(rank) + 1u) * nz) / static_cast<std::size_t>(size);
                picBunch::Field slab;
                slab.grid = field.grid;
                slab.zSlab = true;
                slab.slabOffset = z0;
                slab.slabCount = z1 - z0;
                for(int c = 0; c < 6; ++c)
                    slab.data[c].assign(
                        field.data[c].begin() + static_cast<std::ptrdiff_t>(z0 * plane),
                        field.data[c].begin() + static_cast<std::ptrdiff_t>(z1 * plane));
                picBunch::writeFieldToDomain(
                    slab,
                    *args.output,
                    writeOptions,
                    field.grid.n,
                    field.grid.origin);
            }
            else if(root)
                picBunch::writeField(field, *args.output, writeOptions);
            times.writerMs = elapsedMs(writerStart);
            if(root)
                std::printf(
                    "wrote %s (iteration %llu)\n",
                    args.output->c_str(),
                    static_cast<unsigned long long>(args.outputIteration));
        }
        else if(root)
            std::printf("no --output given; field was computed but not written\n");

        if(root)
            printTimings(times);
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "picBunch: error: %s\n", error.what());
        // After any collective has started a local return would strand the
        // peers; a nonzero abort is the only safe terminal action when more
        // than one rank is running.  Serial runs keep their exit status.
        picBunch::mpiAbort(1);
        return 1;
    }
    catch(...)
    {
        std::fprintf(stderr, "picBunch: error: unknown exception\n");
        picBunch::mpiAbort(1);
        return 1;
    }
    return 0;
}
