// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// picBunch command line: openPMD particles in, free-space E/B out.
// The only accuracy knob is --tolerance; bin counts, interpolation orders and
// FFT sizes are implementation details and are never exposed.

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
            "                        are written as exact zero\n"
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
            "  --near-shell <mode>   none|exact (default: none).  exact replaces the\n"
            "                        mesh contribution of every source within the shell\n"
            "                        by the exact TSC-convolved field; expensive.\n"
            "  --near-shell-radius <cells>  shell radius in PIC cells (default: derived\n"
            "                        from --tolerance: 3 at 1e-3, +1 per decade, max 6)\n"
            "  --near-shell-tile <n> tile edge in nodes for the local subtraction FFT\n"
            "                        (default: 4)\n"
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
        std::string nearShell = "none";
        double nearShellRadius = 0.0;
        std::size_t nearShellTile = 4;
        bool deterministicDeposit = false;
        bool allowFieldTruncation = false;
        std::string backend = "auto";
        std::optional<std::array<double, 3>> cellSize;
        std::optional<std::array<std::size_t, 3>> grid;
        std::optional<std::array<double, 3>> origin;
        std::optional<std::array<std::size_t, 3>> globalGrid;
        std::optional<std::array<double, 3>> globalOrigin;
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
    double parseTolerance(std::string const& text)
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
                "invalid value '" + text + "' for --tolerance: expected a number");
        }
        if(consumed != text.size())
            throw std::invalid_argument(
                "invalid value '" + text + "' for --tolerance: trailing characters");
        return value;
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

    std::array<double, 3> parseTripleDouble(std::string const& text, char const* option)
    {
        auto const parts = splitTriple(text, option);
        std::array<double, 3> result{};
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

    std::array<std::size_t, 3> parseTripleSize(std::string const& text, char const* option)
    {
        auto const parts = splitTriple(text, option);
        std::array<std::size_t, 3> result{};
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
            else if(option == "--near-shell")
                args.nearShell = value("--near-shell");
            else if(option == "--near-shell-radius")
                args.nearShellRadius
                    = parseTripleDouble(value("--near-shell-radius"), "--near-shell-radius")[0];
            else if(option == "--near-shell-tile")
                args.nearShellTile
                    = parseIteration(value("--near-shell-tile"), "--near-shell-tile");
            else if(option == "--deterministic-deposit")
                args.deterministicDeposit = true;
            else if(option == "--allow-field-truncation")
                args.allowFieldTruncation = true;
            else if(option == "--backend")
                args.backend = value("--backend");
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
        if(args.backend != "auto" && args.backend != compiledBackend())
            throw std::invalid_argument(
                "--backend " + args.backend + " is not available; this build uses " + compiledBackend());
        if(args.nearShell != "none" && args.nearShell != "exact")
            throw std::invalid_argument("--near-shell must be 'none' or 'exact'");
        if(!(args.nearShellRadius >= 0.0) || !std::isfinite(args.nearShellRadius))
            throw std::invalid_argument("--near-shell-radius must be finite and non-negative");
        if(args.nearShellTile == 0u)
            throw std::invalid_argument("--near-shell-tile must be positive");
        if(args.restartFormat != "standard" && args.restartFormat != "picongpu")
            throw std::invalid_argument("--restart-format must be 'standard' or 'picongpu'");
        if(args.restartFormat == "picongpu" && args.output)
            throw std::invalid_argument("--restart-format picongpu writes to --output-dir; do not pass --output");
        if(args.cellSize)
            for(double c : *args.cellSize)
                if(!(c > 0.0))
                    throw std::invalid_argument("--cell-size values must be positive metres");
        if(args.origin)
            for(double o : *args.origin)
                if(!std::isfinite(o))
                    throw std::invalid_argument("--origin values must be finite metres");
        if(args.globalOrigin)
            for(double o : *args.globalOrigin)
                if(!std::isfinite(o))
                    throw std::invalid_argument("--global-origin values must be finite metres");
        if(args.globalGrid)
            for(std::size_t n : *args.globalGrid)
                if(n == 0u || n % 2u != 0u)
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
        std::printf("  %-8s %10.3f\n", "readout", times.readoutMs);
        std::printf("  %-8s %10.3f\n", "writer", times.writerMs);
        double total = times.readerMs + times.depositMs + times.kernelMs + times.fftMs
            + times.readoutMs + times.writerMs;
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
        options.tolerance = config.accuracy.tolerance;
        options.cellSize = config.domain.cellSize;
        options.gridSize = config.domain.cells;
        options.gridOrigin = config.domain.origin;
        options.nearShell.mode = (config.accuracy.nearShell == "exact")
            ? picBunch::NearShellOptions::Mode::Exact
            : picBunch::NearShellOptions::Mode::None;
        options.nearShell.radiusCells = config.accuracy.nearShellRadius;
        options.nearShell.tileNodes = config.accuracy.nearShellTile;
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
        restart.float64 = config.picongpu.float64;

        picBunch::GeneratedParticles generated;
        generated.species = config.picongpu.species;
        generated.bunch = placed.bunch;
        generated.domainOrigin = config.domain.origin;
        generated.cellSize = config.domain.cellSize;
        generated.patchOffset = placed.patchOffset;
        generated.patchExtent = placed.patchExtent;
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
    picBunch::StageTimes times;
    try
    {
        picBunch::BunchData bunchData;
        picBunch::Field field;
        std::optional<picBunch::FarTail> farTail;
        bool const root = picBunch::mpiRank() == 0;
        std::array<double, 3> globalOriginResolved{0.0, 0.0, 0.0};

        if(root)
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
        picBunch::ComputeOptions options;
        options.tolerance = args.tolerance;
        options.deterministicDeposit = args.deterministicDeposit;
        options.nearShell.mode = (args.nearShell == "exact")
            ? picBunch::NearShellOptions::Mode::Exact
            : picBunch::NearShellOptions::Mode::None;
        options.nearShell.radiusCells = args.nearShellRadius;
        options.nearShell.tileNodes = args.nearShellTile;
        std::optional<std::array<double, 3>> meshCellSize;
        std::optional<std::array<double, 3>> meshGridOrigin;
        std::optional<std::array<std::size_t, 3>> meshGridExtent;
        try
        {
            auto const geometry = picBunch::readMeshGeometry(args.input);
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
            auto const [shortest, largest] = std::minmax_element(
                options.cellSize.begin(), options.cellSize.end());
            double const ratio = *largest / *shortest;
            if(ratio >= 2.0)
                std::fprintf(
                    stderr,
                    "picBunch: warning: anisotropic cell spacing: largest %.6g m, "
                    "shortest %.6g m (ratio %.3g); near-field accuracy is set by the "
                    "shortest axis (%.6g m)\n",
                    *largest, *shortest, ratio, *shortest);
        }
        // The mesh extent is the default computation grid; --grid is an explicit
        // override.  A --global-grid must not enlarge the computation, so the
        // mesh extent is ignored for the compute box in that case.
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

        // --------------------------------------------------------------- compute
        std::printf(
            "near shell: mode=%s R=%.3g PIC cells tile=%zu nodes\n",
            args.nearShell.c_str(),
            picBunch::resolveNearShellRadius(options, options.nearShell),
            options.nearShell.tileNodes);
        if(options.nearShell.mode == picBunch::NearShellOptions::Mode::Exact)
        {
            std::size_t exactPairs = 0;
            options.nearShell.exactPairs = &exactPairs;
            field = picBunch::computeField(bunchData.bunch, options, &times);
            std::printf("near shell: %zu exact (source,node,component) evaluations\n", exactPairs);
        }
        else
            field = picBunch::computeField(bunchData.bunch, options, &times);

        auto fieldPeaks = [](picBunch::Field const& value) {
            std::array<double, 2> peaks{0.0, 0.0};
            for(int c = 0; c < 6; ++c)
                for(double entry : value.data[c])
                    peaks[c < 3 ? 0 : 1] = std::max(peaks[c < 3 ? 0 : 1], std::abs(entry));
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
            picBunch::FarTail const tail = picBunch::deriveFarTail(bunchData.bunch);
            auto const bounds = bunchData.bunch.bounds();
            std::array<double, 3> tailMargin{};
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
                auto const computeWithMargin = [&](std::array<double, 3> const& useMargin) {
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
                std::array<double, 3> preferred = margin.marginCells;
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

        printGrid(field.grid);
        std::printf("field: max|E|=%g V/m  max|B|=%g T\n", maxE, maxB);
        } // root

        // Rank 0 computed the bunch-box field; every rank needs it to write its
        // own disjoint global chunk.  The transfer is the small bunch box, never
        // the global domain.
        picBunch::mpiBroadcastField(field, 0);
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
            double tailData[23] = {};
            tailData[0] = farTail ? 1.0 : 0.0;
            if(farTail)
            {
                for(int a = 0; a < 3; ++a)
                {
                    tailData[1 + a] = farTail->centre[a];
                    tailData[5 + a] = farTail->dipole[a];
                    tailData[17 + a] = farTail->meanVelocity[a];
                }
                tailData[4] = farTail->charge;
                for(int i = 0; i < 9; ++i)
                    tailData[8 + i] = farTail->quadrupole[i];
                tailData[20] = farTail->supportRadius;
                tailData[21] = farTail->switchDistance;
                tailData[22] = farTail->gammaMax;
            }
            picBunch::mpiBroadcast(tailData, 23, 0);
            if(!root && tailData[0] != 0.0)
            {
                farTail.emplace();
                for(int a = 0; a < 3; ++a)
                {
                    farTail->centre[a] = tailData[1 + a];
                    farTail->dipole[a] = tailData[5 + a];
                    farTail->meanVelocity[a] = tailData[17 + a];
                }
                farTail->charge = tailData[4];
                for(int i = 0; i < 9; ++i)
                    farTail->quadrupole[i] = tailData[8 + i];
                farTail->supportRadius = tailData[20];
                farTail->switchDistance = tailData[21];
                farTail->gammaMax = tailData[22];
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
        return 1;
    }
    return 0;
}
