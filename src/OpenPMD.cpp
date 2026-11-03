// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#include "picBunch/OpenPMD.hpp"

#include <openPMD/openPMD.hpp>

namespace picBunch
{
    OpenPMDInfo openPMDInfo()
    {
        OpenPMDInfo info;
        info.version = openPMD::getVersion();
#if defined(openPMD_HAVE_HDF5) && openPMD_HAVE_HDF5
        info.hdf5 = true;
#else
        info.hdf5 = false;
#endif
        return info;
    }

} // namespace picBunch
