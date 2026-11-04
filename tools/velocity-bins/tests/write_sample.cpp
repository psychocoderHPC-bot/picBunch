// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Deterministic openPMD sample used by tools/verify/check_openpmd.py.
// Writes the same field as tests/test_io.cpp so the Python checker has a small
// known array to validate from outside openPMD-api.

#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "usage: %s <output.h5>\n", argv[0]);
        return 2;
    }

    picBunch::Field field;
    field.grid.n = {4, 5, 6};
    field.grid.spacing = {0.5, 1.5, 2.5};
    field.grid.origin = {1.0, 2.0, 3.0};
    field.grid.fftPadding = 2;
    for(int c = 0; c < 6; ++c)
    {
        field.data[c].resize(field.grid.size());
        for(std::size_t l = 0; l < field.grid.size(); ++l)
            field.data[c][l] = 0.125 * static_cast<double>(c) + 1.0e-3 * static_cast<double>(l)
                + 1.0e-12 * static_cast<double>(c * 7 + l);
    }

    picBunch::FieldWriteOptions options;
    options.iteration = 7u;
    options.time = 2.5e-13;
    options.timeUnitSI = 1.0;
    options.timeOffset = 0.0;
    try
    {
        picBunch::writeField(field, std::string(argv[1]), options);
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "write sample: %s\n", error.what());
        return 1;
    }
    return 0;
}
