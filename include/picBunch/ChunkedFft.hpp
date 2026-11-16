// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace picBunch::detail
{
    /*! Counters and the peak live staging footprint of one staged solve. */
    struct ChunkedSolveStats
    {
        std::uint64_t readBytes = 0;
        std::uint64_t writtenBytes = 0;
        std::uint64_t peakChunkBytes = 0;
        std::uint64_t peakStagingBytes = 0;
        std::uint64_t stagingLiveBytes = 0;
        //! Per-phase measured staging traffic (bytes).
        std::uint64_t sourceWrite = 0;
        std::uint64_t sourceRead = 0;
        std::uint64_t sourceSpecWrite = 0;
        std::uint64_t sourceSpecRead = 0;
        std::uint64_t kernelWrite = 0;
        std::uint64_t kernelRead = 0;
        std::uint64_t backWrite = 0;
        std::uint64_t backRead = 0;
        double fftSeconds = 0.0;
        double transposeSeconds = 0.0;
    };

    /*! One raw double-complex staging file with 64-bit explicit-offset I/O.
     *
     * Every read and write is an explicit chunk, counted for the reported I/O
     * volume.  The file is not mmap'd: the page cache is not a memory bound.
     */
    class StagingFile
    {
    public:
        StagingFile() = default;

        StagingFile(std::string path, ChunkedSolveStats* stats)
            : m_path{std::move(path)}
            , m_stats{stats}
        {
            m_fd = ::open(m_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
            if(m_fd < 0)
                throw std::runtime_error(
                    "staging: cannot create '" + m_path + "': " + std::strerror(errno));
        }

        StagingFile(StagingFile const&) = delete;
        StagingFile& operator=(StagingFile const&) = delete;

        StagingFile(StagingFile&& other) noexcept
            : m_path{std::move(other.m_path)}
            , m_fd{other.m_fd}
            , m_size{other.m_size}
            , m_stats{other.m_stats}
        {
            other.m_fd = -1;
            other.m_size = 0;
        }

        StagingFile& operator=(StagingFile&& other) noexcept
        {
            if(this != &other)
            {
                reset();
                m_path = std::move(other.m_path);
                m_fd = other.m_fd;
                m_size = other.m_size;
                m_stats = other.m_stats;
                other.m_fd = -1;
                other.m_size = 0;
            }
            return *this;
        }

        ~StagingFile()
        {
            reset();
        }

        [[nodiscard]] int fd() const
        {
            return m_fd;
        }
        [[nodiscard]] std::uint64_t sizeBytes() const
        {
            return m_size;
        }

        void readAt(void* destination, std::size_t bytes, std::uint64_t byteOffset)
        {
            auto* out = static_cast<unsigned char*>(destination);
            std::size_t done = 0;
            while(done < bytes)
            {
                std::size_t const chunk = std::min<std::size_t>(bytes - done, kMaxIo);
                ssize_t const got = ::pread(
                    m_fd,
                    out + done,
                    chunk,
                    static_cast<off_t>(byteOffset + done));
                if(got < 0)
                {
                    if(errno == EINTR)
                        continue;
                    throw std::runtime_error(
                        "staging: read '" + m_path + "' failed: " + std::strerror(errno));
                }
                if(got == 0)
                    throw std::runtime_error(
                        "staging: short read from '" + m_path + "' at byte "
                        + std::to_string(byteOffset + done));
                done += static_cast<std::size_t>(got);
            }
            if(m_stats)
            {
                m_stats->readBytes += bytes;
            }
        }

        void writeAt(void const* source, std::size_t bytes, std::uint64_t byteOffset)
        {
            auto const* in = static_cast<unsigned char const*>(source);
            std::size_t done = 0;
            while(done < bytes)
            {
                std::size_t const chunk = std::min<std::size_t>(bytes - done, kMaxIo);
                ssize_t const put = ::pwrite(
                    m_fd,
                    in + done,
                    chunk,
                    static_cast<off_t>(byteOffset + done));
                if(put < 0)
                {
                    if(errno == EINTR)
                        continue;
                    throw std::runtime_error(
                        "staging: write '" + m_path + "' failed: " + std::strerror(errno));
                }
                done += static_cast<std::size_t>(put);
            }
            std::uint64_t const oldSize = m_size;
            m_size = std::max<std::uint64_t>(m_size, byteOffset + bytes);
            if(m_stats)
            {
                m_stats->writtenBytes += bytes;
                m_stats->stagingLiveBytes += m_size - oldSize;
                m_stats->peakStagingBytes
                    = std::max(m_stats->peakStagingBytes, m_stats->stagingLiveBytes);
            }
        }

        void reset()
        {
            if(m_fd >= 0)
            {
                ::close(m_fd);
                m_fd = -1;
                if(!m_keep)
                    ::unlink(m_path.c_str());
                if(m_stats)
                    m_stats->stagingLiveBytes
                        -= std::min(m_stats->stagingLiveBytes, m_size);
            }
            m_size = 0;
        }

        void keep(bool value)
        {
            m_keep = value;
        }

    private:
        static constexpr std::size_t kMaxIo = std::size_t{1} << 30;
        std::string m_path;
        int m_fd = -1;
        std::uint64_t m_size = 0;
        ChunkedSolveStats* m_stats = nullptr;
        bool m_keep = false;
    };

} // namespace picBunch::detail
