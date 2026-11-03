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
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

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
            "  --iteration <N>       output iteration index (default: 0)\n"
            "  --tolerance <value>   accuracy target (default: 1e-3)\n"
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
        std::uint64_t iteration = 0;
        double tolerance = 1e-3;
        std::string backend = "auto";
    };

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
            else if(option == "--iteration")
                args.iteration = std::stoull(value("--iteration"));
            else if(option == "--tolerance")
                args.tolerance = std::stod(value("--tolerance"));
            else if(option == "--backend")
                args.backend = value("--backend");
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
        return args;
    }

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

    void printGrid(picBunch::Grid const& grid)
    {
        std::printf("grid: n=(%zu,%zu,%zu)  spacing=(%.6g,%.6g,%.6g) m\n",
                    grid.n[0], grid.n[1], grid.n[2],
                    grid.spacing[0], grid.spacing[1], grid.spacing[2]);
        std::printf("      origin=(%.6g,%.6g,%.6g) m  fftPadding=%u  fftGrid=(%zu,%zu,%zu)\n",
                    grid.origin[0], grid.origin[1], grid.origin[2], grid.fftPadding,
                    grid.paddedSize()[0], grid.paddedSize()[1], grid.paddedSize()[2]);
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
        selection.iteration = args.iteration;
        if(args.species)
        {
            if(isInteger(*args.species))
                selection.speciesIndex = static_cast<std::size_t>(std::stoull(*args.species));
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

        // The physical cell size is the mesh spacing of the input when it has
        // one; otherwise fall back to one metre and say so.
        picBunch::ComputeOptions options;
        options.tolerance = args.tolerance;
        try
        {
            auto const geometry = picBunch::readMeshGeometry(args.input);
            bool usable = true;
            for(int a = 0; a < 3; ++a)
                usable = usable && geometry.gridSpacing[a] > 0.0
                    && std::isfinite(geometry.gridSpacing[a]);
            if(usable)
            {
                options.cellSize = geometry.gridSpacing;
                std::printf(
                    "cellSize from input mesh E: (%.6g,%.6g,%.6g) m\n",
                    options.cellSize[0], options.cellSize[1], options.cellSize[2]);
            }
            else
                std::printf("cellSize: input mesh geometry is unusable, using 1 m\n");
        }
        catch(std::exception const&)
        {
            std::printf("cellSize: input has no readable mesh, using 1 m\n");
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
        if(args.output)
        {
            picBunch::FieldWriteOptions writeOptions;
            writeOptions.iteration = args.iteration;
            writeOptions.time = bunchData.time;
            writeOptions.timeUnitSI = bunchData.timeUnitSI;
            auto const writerStart = Clock::now();
            picBunch::writeField(field, *args.output, writeOptions);
            times.writerMs = elapsedMs(writerStart);
            std::printf(
                "wrote %s (iteration %llu)\n",
                args.output->c_str(),
                static_cast<unsigned long long>(args.iteration));
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
