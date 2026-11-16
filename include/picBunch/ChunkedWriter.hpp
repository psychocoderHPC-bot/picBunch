// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include "picBunch/OpenPMDWriter.hpp"
#include "picBunch/Types.hpp"

#include <memory>
#include <string>

namespace picBunch
{
    /** Stream z-slabs of the six field components into one openPMD series.
     *
     * The series and every E/B dataset are created once with the full global
     * extent, exactly as ``writeField``/``writeFieldToDomain`` do.  Each call to
     * ``write`` stores one component slab at its global offset.  The datasets
     * therefore have the same extent and component metadata as the in-core
     * writer, while peak memory is one slab.
     *
     * Every component must be covered exactly once before ``close`` succeeds.
     * The series is written under a temporary name and renamed only after a
     * successful close, so a failed chunked write leaves no partial output.
     */
    class ChunkedFieldWriter
    {
    public:
        ChunkedFieldWriter(
            std::string fileName,
            FieldWriteOptions options,
            Idx3 globalGridSize,
            Vec3 globalGridOrigin,
            Grid grid);
        ~ChunkedFieldWriter();

        ChunkedFieldWriter(ChunkedFieldWriter const&) = delete;
        ChunkedFieldWriter& operator=(ChunkedFieldWriter const&) = delete;

        //! Store one component z-slab.  Throws if a plane is stored twice.
        void write(FieldChunk const& chunk);
        //! Verify full coverage, flush, close and publish the series.
        void close();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace picBunch
