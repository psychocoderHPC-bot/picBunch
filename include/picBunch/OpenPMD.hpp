// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <string>

namespace picBunch
{
    /** Build-time/runtime openPMD-api information.
     *
     * The reader and writer stages require the HDF5 backend; CMake rejects a
     * configuration whose openPMD-api was built without it.
     */
    struct OpenPMDInfo
    {
        std::string version;
        bool hdf5 = false;
    };

    OpenPMDInfo openPMDInfo();

} // namespace picBunch
