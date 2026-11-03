// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// picBunch command line: openPMD particles in, free-space E/B out.
// The only accuracy knob is --tolerance; bin counts, interpolation orders and
// FFT sizes are implementation details and are never exposed.

#include "picBunch/Compute.hpp"
#include "picBunch/OpenPMD.hpp"
#include "picBunch/OpenPMDWriter.hpp"
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
            "Usage: picBunch --input <openPMD file/series> [options]\n"
            "\n"
            "Options:\n"
            "  --input <path>        openPMD particle series (required)\n"
            "  --species <name|idx>  particle species (default: first)\n"
            "  --output <path>       write E/B as an openPMD series\n"
            "  --input-iteration <N> input iteration index (default: first)\n"
            "  --output-iteration <N> output iteration index (default: 0)\n"
            "  --cell-size <dx>[,<dy>[,<dz>]]\n"
            "                        physical cell size in metres (overrides the\n"
            "                        input mesh; required when there is no mesh)\n"
            "  --grid <nx>[,<ny>[,<nz>]]  explicit output node counts (even)\n"
            "  --origin <x>[,<y>[,<z>]]   explicit output grid origin in metres\n"
            "  --restart-format <name>    standard|picongpu (default: standard)\n"
            "  --output-dir <dir>    PIConGPU checkpoint directory (default: checkpoints)\n"
            "  --picongpu-unit-length <m>   length unit for the restart writer\n"
            "  --picongpu-unit-e <V/m>      E-field unit for the restart writer\n"
            "  --picongpu-unit-b <T>        B-field unit for the restart writer\n"
            "  --picongpu-step <N>          checkpoint step (default: output iteration)\n"
            "  --picongpu-species <name>    particle species to copy into the checkpoint\n"
            "  --picongpu-dt <value>        stored dt attribute (default: 0)\n"
            "  --tolerance <value>   relative accuracy target of the free-space\n"
            "                        convolution (default: 1e-3; floor 1e-14)\n"
            "  --backend <name>      auto|cuda|hip|cpu (default: auto)\n"
            "  --help                print this help\n"
            "\n"
            "Compiled backend: %s\n",
            compiledBackend());
    }

    struct Arguments
    {
        std::string input;
        std::optional<std::string> species;
        std::optional<std::string> output;
        std::optional<std::uint64_t> inputIteration;
        std::uint64_t outputIteration = 0;
        double tolerance = 1e-3;
        std::string backend = "auto";
        std::optional<std::array<double, 3>> cellSize;
        std::optional<std::array<std::size_t, 3>> grid;
        std::optional<std::array<double, 3>> origin;
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
            else if(option == "--backend")
                args.backend = value("--backend");
            else if(option == "--cell-size")
                args.cellSize = parseTripleDouble(value("--cell-size"), "--cell-size");
            else if(option == "--grid")
                args.grid = parseTripleSize(value("--grid"), "--grid");
            else if(option == "--origin")
                args.origin = parseTripleDouble(value("--origin"), "--origin");
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
        if(!(args.tolerance > 0.0) || !std::isfinite(args.tolerance))
            throw std::invalid_argument("--tolerance must be finite and positive");
        if(args.backend != "auto" && args.backend != compiledBackend())
            throw std::invalid_argument(
                "--backend " + args.backend + " is not available; this build uses " + compiledBackend());
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

    picBunch::StageTimes times;
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
        picBunch::BunchData const bunchData = picBunch::readBunch(args.input, selection);
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
        std::optional<std::array<double, 3>> meshCellSize;
        try
        {
            auto const geometry = picBunch::readMeshGeometry(args.input);
            // readMeshGeometry already returns physical SI spacing and offset
            // (raw gridSpacing/gridGlobalOffset times gridUnitSI).  Applying
            // gridUnitSI again here is the v0.1 regression; the reader owns the
            // conversion and this helper only validates the result.
            meshCellSize = picBunch::physicalCellSize(geometry);
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
        if(args.grid)
        {
            options.gridSize = args.grid;
            std::printf(
                "grid: --grid override (%zu,%zu,%zu) nodes\n",
                (*args.grid)[0], (*args.grid)[1], (*args.grid)[2]);
        }
        if(args.origin)
        {
            options.gridOrigin = args.origin;
            std::printf(
                "origin: --origin override (%.6g,%.6g,%.6g) m\n",
                (*args.origin)[0], (*args.origin)[1], (*args.origin)[2]);
        }

        // --------------------------------------------------------------- compute
        picBunch::Field const field = picBunch::computeField(bunchData.bunch, options, &times);
        printGrid(field.grid);

        double maxE = 0.0;
        double maxB = 0.0;
        for(int c = 0; c < 6; ++c)
            for(double value : field.data[c])
            {
                if(c < 3)
                    maxE = std::max(maxE, std::abs(value));
                else
                    maxB = std::max(maxB, std::abs(value));
            }
        std::printf("field: max|E|=%g V/m  max|B|=%g T\n", maxE, maxB);

        // ---------------------------------------------------------------- writer
        if(args.restartFormat == "picongpu")
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
            restart.nextId = bunchData.bunch.size();

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
            picBunch::writeField(field, *args.output, writeOptions);
            times.writerMs = elapsedMs(writerStart);
            std::printf(
                "wrote %s (iteration %llu)\n",
                args.output->c_str(),
                static_cast<unsigned long long>(args.outputIteration));
        }
        else
            std::printf("no --output given; field was computed but not written\n");

        printTimings(times);
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "picBunch: error: %s\n", error.what());
        return 1;
    }
    return 0;
}
