// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <cstddef>

namespace picBunch
{
    namespace detail
    {
        //! Default cap of the process-wide kernel-spectrum cache, in MiB.
        inline constexpr std::size_t kDefaultFftSpectrumCacheMiB = 1536u;

        //! Environment variable that overrides the built-in cap, in MiB. 0 disables the cache.
        inline constexpr char const* kFftSpectrumCacheEnv = "PICBUNCH_FFT_SPECTRUM_CACHE_MB";

        /*! Resolve the active cache cap in MiB.
         *
         * Precedence: explicit ``setFftSpectrumCacheLimitMiB`` (the CLI option) wins over
         * ``PICBUNCH_FFT_SPECTRUM_CACHE_MB``, which wins over the 1536 MiB default. A value
         * of 0 disables the cache. The environment value is read on every resolution so a
         * caller can change it in-process; an invalid value throws ``std::invalid_argument``.
         */
        std::size_t fftSpectrumCacheLimitMiB();

        //! Active cache cap in bytes. 0 means disabled.
        std::size_t fftSpectrumCacheLimitBytes();

        //! Process-wide override, used by ``--fft-spectrum-cache-mb``. 0 disables the cache.
        void setFftSpectrumCacheLimitMiB(std::size_t mib);

        //! Drop the process-wide override so the environment/default applies again.
        void clearFftSpectrumCacheLimitOverride();

        //! Snapshot of the process-wide cache, for tests and diagnostics.
        struct FftSpectrumCacheStats
        {
            std::size_t entries = 0u;
            std::size_t bytes = 0u;
            std::size_t hits = 0u;
            std::size_t misses = 0u;
        };

        FftSpectrumCacheStats fftSpectrumCacheStats();
    } // namespace detail
} // namespace picBunch
