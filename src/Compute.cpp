// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Compute core of picBunch: TSC deposit, TSC-convolved moving point kernel,
// free-space convolution through the alpakaVendor FFT, Yee read-out.
//
// All precision-critical stages are double.  The FFT is unnormalized, so the
// caller divides by M^3; the mesh density is charge-per-node, so the physical
// field additionally carries h^3 (h = nodes per cell).  The delivered reference
// implementation forgot h^3 because it only ran h = 1; here scale = h^3 / M^3.

#include "picBunch/Compute.hpp"

#include "picBunch/Deposit.hpp"
#include "picBunch/Kernel.hpp"
#include "picBunch/NearShell.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
#    include <rocfft/rocfft.h>
#endif
#include <alpaka/math/Complex.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace picBunch
{
    namespace
    {
        using Extents3 = alpaka::Vec<std::uint32_t, 3u>;
        using Complex = alpaka::math::Complex<double>;

        template<typename T_Vec>
        T_Vec ceilDivVec(T_Vec const& a, T_Vec const& b)
        {
            T_Vec r{};
            for(std::uint32_t i = 0; i < T_Vec::dim(); ++i)
                r[i] = static_cast<decltype(a[0])>((a[i] + b[i] - 1u) / b[i]);
            return r;
        }

        //! TSC assignment of weight*charge onto the n^3 sub-box of the M^3 padded grid.
        struct TscDepositKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double const* px,
                double const* py,
                double const* pz,
                double const* pq,
                std::uint32_t numParticles,
                double* rho,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                double ox,
                double oy,
                double oz,
                double sx,
                double sy,
                double sz) const
            {
                double const support = 1.5;
                for(auto [p] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{numParticles}))
                {
                    double const pos[3] = {px[p], py[p], pz[p]};
                    double const org[3] = {ox, oy, oz};
                    double const sp[3] = {sx, sy, sz};
                    double t[3];
                    long lo[3];
                    long hi[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        t[a] = (pos[a] - org[a]) / sp[a];
                        lo[a] = static_cast<long>(::ceil(t[a] - support));
                        hi[a] = static_cast<long>(::floor(t[a] + support));
                    }
                    double const q = pq[p];
                    for(long ia = lo[0]; ia <= hi[0]; ++ia)
                    {
                        if(ia < 0 || ia >= static_cast<long>(nx))
                            continue;
                        double const wa = tscShape(static_cast<double>(ia) - t[0]);
                        if(wa == 0.0)
                            continue;
                        for(long ib = lo[1]; ib <= hi[1]; ++ib)
                        {
                            if(ib < 0 || ib >= static_cast<long>(ny))
                                continue;
                            double const wb = wa * tscShape(static_cast<double>(ib) - t[1]);
                            if(wb == 0.0)
                                continue;
                            for(long ic = lo[2]; ic <= hi[2]; ++ic)
                            {
                                if(ic < 0 || ic >= static_cast<long>(nz))
                                    continue;
                                double const wc = wb * tscShape(static_cast<double>(ic) - t[2]);
                                if(wc == 0.0)
                                    continue;
                                auto const idx = static_cast<std::size_t>(ia)
                                    + static_cast<std::size_t>(Mx)
                                        * (static_cast<std::size_t>(ib)
                                           + static_cast<std::size_t>(My) * static_cast<std::size_t>(ic));
                                alpaka::onAcc::atomicAdd(acc, &rho[idx], wc * q);
                            }
                        }
                    }
                }
            }
        };

        //! One component of the TSC-convolved moving point kernel on the padded box.
        struct KernelBuildKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* kernel,
                std::uint32_t Mx,
                std::uint32_t My,
                std::uint32_t Mz,
                int component,
                double dx,
                double dy,
                double dz,
                double cellx,
                double celly,
                double cellz,
                double refinement,
                double ux,
                double uy,
                double uz) const
            {
                for(auto const& idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{Extents3{Mx, My, Mz}}))
                {
                    auto const i = static_cast<std::uint32_t>(idx[0]);
                    auto const j = static_cast<std::uint32_t>(idx[1]);
                    auto const k = static_cast<std::uint32_t>(idx[2]);
                    // Centred index of a node on the (zero-padded) FFT box.  ``i <= (M-1)/2``
                    // is the parity-correct split: for even M it is identical to ``i < M/2``
                    // (bins 0..M/2-1 positive, M/2..M-1 negative, Nyquist bin at M/2); for odd
                    // M (no Nyquist plane) it is 0..(M-1)/2 positive and (M+1)/2..M-1 negative.
                    // A plain ``i < M/2`` mis-centres the kernel on an odd extent.
                    int const di = (i <= (Mx - 1u) / 2u) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(Mx);
                    int const dj = (j <= (My - 1u) / 2u) ? static_cast<int>(j) : static_cast<int>(j) - static_cast<int>(My);
                    int const dk = (k <= (Mz - 1u) / 2u) ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(Mz);
                    // R = (d*h + yee_delta) * cell
                    double const Rx = (static_cast<double>(di) * refinement + dx) * cellx;
                    double const Ry = (static_cast<double>(dj) * refinement + dy) * celly;
                    double const Rz = (static_cast<double>(dk) * refinement + dz) * cellz;
                    double field[6];
                    pointFieldNormalized(Rx, Ry, Rz, ux, uy, uz, field);
                    double value = field[component];
                    if(!std::isfinite(value))
                        value = 0.0;
                    auto const flat = static_cast<std::size_t>(i)
                        + static_cast<std::size_t>(Mx)
                            * (static_cast<std::size_t>(j)
                               + static_cast<std::size_t>(My) * static_cast<std::size_t>(k));
                    kernel[flat] = value;
                }
            }
        };

        //! Multiply two packed half-spectra.
        //!
        //! The product of two half-spectra of real arrays is the half-spectrum of a real
        //! array, so the packed planes are already consistent for C2R.  Do not zero the
        //! imaginary part of the last-axis DC/Nyquist planes: in 3-D R2C those planes are
        //! a complex 2-D transform, not a real 1-D transform, and zeroing them injects a
        //! constant field error that grows with the kernel's dynamic range.
        struct MultiplySpectrumKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* a,
                double const* b,
                std::uint32_t count) const
            {
                for(auto [l] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{count}))
                {
                    std::size_t const off = 2u * static_cast<std::size_t>(l);
                    double const ar = a[off];
                    double const ai = a[off + 1u];
                    double const br = b[off];
                    double const bi = b[off + 1u];
                    double re = ar * br - ai * bi;
                    double im = ar * bi + ai * br;
                    a[off] = re;
                    a[off + 1u] = im;
                }
            }
        };

        using Clock = std::chrono::steady_clock;

        double elapsedMs(Clock::time_point const& start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        /*! Measured relative error of the padding-2 result.
         *
         * The support of the free-space convolution is ``n`` output nodes; padding
         * 2 makes the FFT box ``2n`` wide, so every source-observer lag is shorter
         * than half the box and no periodic image can reach an observer.  The
         * remaining difference to padding 3 is FFT/roundoff only.  Measured on
         * rosi_v100 (CUDA, V100) as ``max|pad2-pad3| / max|field|`` for the toy
         * lattice used by the padding test: 8.37e-16; a 500-particle
         * ``cellSize = 1e-6 m`` cloud on an 18^3 grid gives 4.15e-16.  This value
         * is a conservative bound over both, rounded up by roughly three orders
         * of magnitude to cover geometries and backends that were not measured.
         */
        //! Largest absolute eigenvalue of a symmetric 3x3 matrix (row-major).
        double quadrupoleSpectralNorm(std::array<double, 9> const& a)
        {
            double const p1 = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
            if(p1 == 0.0)
                return std::max({std::abs(a[0]), std::abs(a[4]), std::abs(a[8])});
            double const q = (a[0] + a[4] + a[8]) / 3.0;
            double const p2 = (a[0] - q) * (a[0] - q) + (a[4] - q) * (a[4] - q)
                + (a[8] - q) * (a[8] - q) + 2.0 * p1;
            double const p = std::sqrt(p2 / 6.0);
            std::array<double, 9> b{};
            for(int i = 0; i < 9; ++i)
                b[i] = a[i] / p;
            b[0] -= q / p;
            b[4] -= q / p;
            b[8] -= q / p;
            double const det = b[0] * (b[4] * b[8] - b[5] * b[7])
                - b[1] * (b[3] * b[8] - b[5] * b[6])
                + b[2] * (b[3] * b[7] - b[4] * b[6]);
            double const r = std::max(-1.0, std::min(1.0, det / 2.0));
            double const phi = std::acos(r) / 3.0;
            double const e1 = q + 2.0 * p * std::cos(phi);
            double const e3 = q + 2.0 * p * std::cos(phi + 2.0943951023931953);
            double const e2 = 3.0 * q - e1 - e3;
            return std::max({std::abs(e1), std::abs(e2), std::abs(e3)});
        }

        constexpr double kPadding2ErrorBound = 1.0e-12;

        /*! Lowest tolerance the double-precision FFT pipeline can honour.
         *
         * Padding 3 is converged to FFT roundoff (measured 8.37e-16 relative),
         * but accumulation and transform roundoff in a single double run cannot
         * be guaranteed below this floor.  Requests below it are rejected instead
         * of being silently under-delivered.
         */
        constexpr double kMinTolerance = 1.0e-14;

        /*! Hard limits used to turn an infeasible input into a clear error.
         *
         * The v0.1 double conversion made a 1e-6 m cell look like 1e-12 m and
         * the node count ran into ``std::vector``'s max_size, surfacing as a
         * ``bad_alloc``-style message.  A grid beyond these bounds cannot be
         * allocated by the pipeline, so reject it before any allocation.
         */
        constexpr std::size_t kMaxNodesPerAxis = 1u << 20;
        constexpr std::size_t kMaxPaddedNodes = 1u << 28;

        /*! Tolerance -> FFT padding.  Padding 1 is never selected: it is a
         * diagnostic mode whose periodic-image artefact (measured 4.0e-2 for the
         * padding-test geometry) no tolerance can bound.
         */
        unsigned selectFftPadding(double tolerance)
        {
            if(!(tolerance > 0.0) || !std::isfinite(tolerance))
                throw std::invalid_argument("makeGrid: tolerance must be finite and positive");
            if(tolerance < kMinTolerance)
                throw std::runtime_error(
                    "makeGrid: tolerance " + std::to_string(tolerance)
                    + " is below the measured double-precision floor "
                    + std::to_string(kMinTolerance)
                    + "; raise --tolerance (the FFT padding cannot deliver a tighter target)");
            return (tolerance < kPadding2ErrorBound) ? 3u : 2u;
        }
    } // namespace

    std::array<double, 3> resolveCellSize(
        std::optional<std::array<double, 3>> const& explicitCellSize,
        std::optional<std::array<double, 3>> const& meshCellSize)
    {
        auto validate = [](std::array<double, 3> const& value, char const* source) {
            for(double c : value)
                if(!(c > 0.0) || !std::isfinite(c))
                    throw std::invalid_argument(
                        std::string("resolveCellSize: ") + source + " cell size must be finite and positive");
        };
        if(explicitCellSize)
        {
            validate(*explicitCellSize, "--cell-size");
            return *explicitCellSize;
        }
        if(meshCellSize)
        {
            validate(*meshCellSize, "mesh-derived");
            return *meshCellSize;
        }
        throw std::runtime_error(
            "no cell size available: the input has no usable mesh geometry and --cell-size was not given; "
            "pass --cell-size <dx>[,<dy>,<dz>] in metres");
    }

    std::optional<std::array<double, 3>> physicalCellSize(MeshGeometry const& geometry)
    {
        // readMeshGeometry returns physical SI spacing and offset.  Do not
        // multiply by geometry.gridUnitSI here: the reader already did it, and
        // the CLI did it again in v0.1 (physical 1e-6 m became 1e-12 m).
        for(double spacing : geometry.gridSpacing)
            if(!(spacing > 0.0) || !std::isfinite(spacing))
                return std::nullopt;
        return geometry.gridSpacing;
    }

    std::optional<std::array<double, 3>> physicalGridOrigin(MeshGeometry const& geometry)
    {
        // Same ownership rule as the spacing: readMeshGeometry already multiplied
        // gridGlobalOffset by gridUnitSI, so the CLI must not apply it again.
        for(double offset : geometry.gridGlobalOffset)
            if(!std::isfinite(offset))
                return std::nullopt;
        return geometry.gridGlobalOffset;
    }

    std::optional<std::array<std::size_t, 3>> physicalGridExtent(MeshGeometry const& geometry)
    {
        // readMeshGeometry already mapped the dataset extent from the stored axis
        // order to x,y,z.  A node count has no unit conversion; the reader owns
        // the gridUnitSI handling of spacing and offset.
        for(std::size_t nodes : geometry.shape)
            if(nodes == 0u)
                return std::nullopt;
        return geometry.shape;
    }

    FieldTailMargin deriveFieldTailMargin(
        Bunch const& bunch,
        double fieldPeak,
        double tolerance,
        std::array<double, 3> const& cellSize,
        double maxDistance)
    {
        if(bunch.size() == 0u)
            throw std::runtime_error("deriveFieldTailMargin: empty bunch");
        if(!(tolerance > 0.0) || !std::isfinite(tolerance))
            throw std::invalid_argument("deriveFieldTailMargin: tolerance must be finite and positive");
        if(!(fieldPeak >= 0.0) || !std::isfinite(fieldPeak))
            throw std::invalid_argument("deriveFieldTailMargin: field peak must be finite and non-negative");
        if(!(maxDistance > 0.0) || !std::isfinite(maxDistance))
            throw std::invalid_argument("deriveFieldTailMargin: maxDistance must be finite and positive");
        for(double h : cellSize)
            if(!(h > 0.0) || !std::isfinite(h))
                throw std::invalid_argument("deriveFieldTailMargin: cell sizes must be finite and positive");

        // Source moments about the bounding-box centre.  Using the box half
        // extents for the per-axis conversion is tighter than the circumscribed
        // radius and still bounds every point outside the expanded box.
        auto const bounds = bunch.bounds();
        std::array<double, 3> centre{};
        std::array<double, 3> halfExtent{};
        for(int a = 0; a < 3; ++a)
        {
            centre[a] = 0.5 * (bounds[0][a] + bounds[1][a]);
            halfExtent[a] = 0.5 * (bounds[1][a] - bounds[0][a]);
        }

        double totalCharge = 0.0;
        double absoluteCharge = 0.0;
        std::array<double, 3> dipole{0.0, 0.0, 0.0};
        std::array<double, 9> quadrupole{};
        double sourceRadius = 0.0;
        double maxGamma = 1.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
        {
            double const u2 = bunch.ux[p] * bunch.ux[p]
                + bunch.uy[p] * bunch.uy[p] + bunch.uz[p] * bunch.uz[p];
            maxGamma = std::max(maxGamma, std::sqrt(1.0 + u2));
            double const q = bunch.weight[p] * bunch.charge[p];
            totalCharge += q;
            absoluteCharge += std::abs(q);
            std::array<double, 3> const displacement{
                bunch.x[p] - centre[0],
                bunch.y[p] - centre[1],
                bunch.z[p] - centre[2]};
            for(int a = 0; a < 3; ++a)
                dipole[a] += q * displacement[a];
            double const r2 = displacement[0] * displacement[0]
                + displacement[1] * displacement[1]
                + displacement[2] * displacement[2];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    quadrupole[i * 3 + j]
                        += q * (3.0 * displacement[i] * displacement[j]
                                - (i == j ? r2 : 0.0));
            sourceRadius = std::max(sourceRadius, std::sqrt(r2));
        }
        double const dipoleMagnitude = std::sqrt(
            dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]);
        double const quadrupoleNorm = quadrupoleSpectralNorm(quadrupole);

        FieldTailMargin result;
        result.totalCharge = totalCharge;
        result.absoluteCharge = absoluteCharge;
        result.dipoleMagnitude = dipoleMagnitude;
        result.dipoleMoment = dipole;
        result.quadrupoleMoment = quadrupole;
        result.quadrupoleNorm = quadrupoleNorm;
        result.sourceRadius = sourceRadius;
        result.maxGamma = maxGamma;
        result.fieldPeak = fieldPeak;
        result.tolerance = tolerance;
        result.distance = sourceRadius;
        result.bound = 0.0;

        double const target = tolerance * fieldPeak;
        if(absoluteCharge == 0.0 || target == 0.0)
            return result; // no source or no field: zero margin is exact

        auto const tailBound = [&](double d) {
            if(!(d > sourceRadius))
                return std::numeric_limits<double>::infinity();
            // Monopole, dipole and the actual quadrupole.  The angular maximum
            // of the quadrupole field is (k/2)(5|S| + 2|D n|)/d^4 with
            // |S|, |D n| <= ||D||_2, hence the factor 3.5.
            return maxGamma * kCoulombConstant
                * (std::abs(totalCharge) / (d * d) + 2.0 * dipoleMagnitude / (d * d * d)
                   + 3.5 * quadrupoleNorm / (d * d * d * d));
        };

        double low = sourceRadius;
        if(low <= 0.0)
            low = 1.0e-6 * std::min({cellSize[0], cellSize[1], cellSize[2]});
        else
            low += 1.0e-12 * std::max(low, std::min({cellSize[0], cellSize[1], cellSize[2]}));
        if(tailBound(low) <= target)
        {
            result.distance = low;
            result.bound = tailBound(low);
        }
        else
        {
            double high = std::max(2.0 * low, sourceRadius + std::min({cellSize[0], cellSize[1], cellSize[2]}));
            while(tailBound(high) > target && high < maxDistance)
                high = std::min(2.0 * high, maxDistance);
            if(tailBound(high) > target)
                throw std::runtime_error(
                    "deriveFieldTailMargin: a neglected-field bound below tolerance requires a distance"
                    " larger than " + std::to_string(maxDistance) + " m; enlarge the compute box or"
                    " acknowledge the truncation explicitly");
            for(int iteration = 0; iteration < 200; ++iteration)
            {
                double const mid = 0.5 * (low + high);
                if(tailBound(mid) > target)
                    low = mid;
                else
                    high = mid;
            }
            result.distance = high;
            result.bound = tailBound(high);
        }

        for(int a = 0; a < 3; ++a)
        {
            double const cells = (result.distance - halfExtent[a]) / cellSize[a];
            result.marginCells[a] = std::max(0.0, std::ceil(cells));
        }
        return result;
    }

    Grid makeGrid(Bunch const& bunch, ComputeOptions const& options)
    {
        if(bunch.size() == 0)
            throw std::runtime_error("makeGrid: empty bunch");
        if(options.marginCells < 0.0)
            throw std::invalid_argument("makeGrid: marginCells must not be negative");
        if(options.marginCellsPerAxis)
            for(double margin : *options.marginCellsPerAxis)
                if(!(margin >= 0.0) || !std::isfinite(margin))
                    throw std::invalid_argument(
                        "makeGrid: every marginCellsPerAxis value must be finite and non-negative");
        for(double c : options.cellSize)
            if(!(c > 0.0) || !std::isfinite(c))
                throw std::invalid_argument("makeGrid: every cell size must be finite and positive");

        auto const b = bunch.bounds();
        Grid grid;
        // The tolerance policy picks the FFT padding; a non-zero fftPadding is an
        // explicit diagnostic override (1 is the wraparound control).
        unsigned const padding = (options.fftPadding != 0u) ? options.fftPadding
                                                           : selectFftPadding(options.tolerance);
        if(padding == 0u)
            throw std::invalid_argument("makeGrid: fftPadding must be >= 1");
        grid.fftPadding = padding;

        for(int a = 0; a < 3; ++a)
        {
            double const spacing = kRefinement * options.cellSize[a];
            double const marginCellCount = options.marginCellsPerAxis
                ? (*options.marginCellsPerAxis)[a]
                : options.marginCells;
            double const margin = marginCellCount * options.cellSize[a];
            double const low = b[0][a] - margin;
            double const high = b[1][a] + margin;
            double const span = (high - low) / spacing;
            if(!std::isfinite(span) || span > static_cast<double>(kMaxNodesPerAxis))
                throw std::runtime_error(
                    "makeGrid: the physical cell size produces " + std::to_string(span)
                    + " nodes on one axis, which is infeasible (limit "
                    + std::to_string(kMaxNodesPerAxis)
                    + "); check the input gridSpacing/gridUnitSI or pass --cell-size");
            std::size_t nodes = static_cast<std::size_t>(std::ceil(span)) + 1u;
            // The auto-derived grid keeps an even node count for backwards
            // compatibility (earlier tests expect even n).  Odd extents are now
            // supported by the R2C path (parity-correct centring in
            // KernelBuildKernel and the relaxed --grid check); this rounding only
            // constrains the derived grid, never an explicit one.
            if(nodes % 2u)
                ++nodes;
            nodes = std::max<std::size_t>(nodes, 8u);
            grid.n[a] = nodes;
            grid.spacing[a] = spacing;
            grid.origin[a] = low;
        }
        // An explicit CLI --grid overrides the mesh extent, which in turn
        // overrides the bunch-bounding-box derivation.  Values are used
        // verbatim and never blended.
        if(options.gridSize)
        {
            for(int a = 0; a < 3; ++a)
            {
                std::size_t const n = (*options.gridSize)[a];
                if(n == 0u)
                    throw std::invalid_argument(
                        "makeGrid: --grid values must be positive; got "
                        + std::to_string((*options.gridSize)[0]) + ","
                        + std::to_string((*options.gridSize)[1]) + ","
                        + std::to_string((*options.gridSize)[2]));
                grid.n[a] = n;
            }
        }
        else if(options.meshGridSize)
        {
            // The mesh extent is the node count of its dataset, already mapped
            // to x,y,z by readMeshGeometry.  Copy it verbatim.
            for(int a = 0; a < 3; ++a)
            {
                if((*options.meshGridSize)[a] == 0u)
                    throw std::invalid_argument("makeGrid: the input mesh extent must be positive");
                grid.n[a] = (*options.meshGridSize)[a];
            }
        }
        // An explicit --origin overrides the mesh gridGlobalOffset, which in
        // turn overrides the derived bunch-box minimum.
        std::optional<std::array<double, 3>> const& originOverride
            = options.gridOrigin ? options.gridOrigin : options.meshGridOrigin;
        if(originOverride)
        {
            for(int a = 0; a < 3; ++a)
            {
                if(!std::isfinite((*originOverride)[a]))
                    throw std::invalid_argument("makeGrid: grid origin values must be finite");
                grid.origin[a] = (*originOverride)[a];
            }
        }

        // A global output domain does not enlarge the computation.  It only
        // says where the small compute box sits inside the written domain.  Snap
        // the compute box onto global nodes so the read-out maps one-to-one and
        // no interpolation is needed; clip it to the global extent.
        if(options.globalGridSize)
        {
            std::array<std::size_t, 3> const globalN = *options.globalGridSize;
            std::array<double, 3> const globalOrigin
                = options.globalGridOrigin.value_or(std::array<double, 3>{0.0, 0.0, 0.0});
            for(int a = 0; a < 3; ++a)
            {
                if(globalN[a] == 0u || globalN[a] % 2u != 0u)
                    throw std::invalid_argument(
                        "makeGrid: --global-grid values must be even and positive");
                if(!std::isfinite(globalOrigin[a]))
                    throw std::invalid_argument("makeGrid: --global-origin values must be finite");
            }
            for(int a = 0; a < 3; ++a)
            {
                double const spacing = grid.spacing[a];
                // Smallest global node at or below the compute-box origin.
                long long low = std::max<long long>(
                    0,
                    static_cast<long long>(
                        std::floor((grid.origin[a] - globalOrigin[a]) / spacing)));
                long long high = std::min<long long>(
                    static_cast<long long>(globalN[a]),
                    static_cast<long long>(std::ceil(
                        (grid.origin[a] + static_cast<double>(grid.n[a]) * spacing
                         - globalOrigin[a])
                        / spacing)));
                if(low >= high)
                    throw std::runtime_error(
                        "makeGrid: the compute box does not overlap the global output domain on axis "
                        + std::to_string(a));
                // Keep the R2C last extent even.  Prefer growing, then shrinking
                // the low side, so a particle at the upper edge is never dropped.
                if((high - low) % 2 != 0)
                {
                    if(high < static_cast<long long>(globalN[a]))
                        ++high;
                    else
                        --low;
                }
                grid.origin[a] = globalOrigin[a] + static_cast<double>(low) * spacing;
                grid.n[a] = static_cast<std::size_t>(high - low);
            }
        }

        // The TSC deposit silently skips nodes outside the output grid, so a
        // grid that does not contain every particle would drop charge without
        // a trace.  Reject that instead.  The physical extent of the node set
        // is [origin, origin + n*spacing], matching the mesh dataset extent.
        {
            std::array<double, 3> lower{};
            std::array<double, 3> upper{};
            for(int a = 0; a < 3; ++a)
            {
                lower[a] = grid.origin[a];
                upper[a] = grid.origin[a] + static_cast<double>(grid.n[a]) * grid.spacing[a];
            }
            std::size_t outside = 0u;
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                double const position[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                for(int a = 0; a < 3; ++a)
                {
                    if(position[a] < lower[a] || position[a] > upper[a])
                    {
                        ++outside;
                        break;
                    }
                }
            }
            if(outside != 0u)
                throw std::runtime_error(
                    "makeGrid: " + std::to_string(outside) + " of " + std::to_string(bunch.size())
                    + " particles lie outside the output grid bounds "
                    + "[" + std::to_string(lower[0]) + ", " + std::to_string(upper[0]) + "] x "
                    + "[" + std::to_string(lower[1]) + ", " + std::to_string(upper[1]) + "] x "
                    + "[" + std::to_string(lower[2]) + ", " + std::to_string(upper[2]) + "]"
                    + "; the grid comes from the input mesh or --grid/--origin/--cell-size,"
                    + " and out-of-grid particles would be dropped");
        }

        // Reject an infeasible grid before the first allocation.  Both the
        // per-axis and the padded total are checked; the product is formed in
        // 64-bit with an explicit overflow guard.
        std::uint64_t padded = 1u;
        for(int a = 0; a < 3; ++a)
        {
            if(grid.n[a] > kMaxNodesPerAxis)
                throw std::runtime_error(
                    "makeGrid: output grid " + std::to_string(grid.n[0]) + "x"
                    + std::to_string(grid.n[1]) + "x" + std::to_string(grid.n[2])
                    + " is infeasible (per-axis limit " + std::to_string(kMaxNodesPerAxis)
                    + "); check the input gridSpacing/gridUnitSI or pass --cell-size");
            std::uint64_t const axisPadded = static_cast<std::uint64_t>(grid.n[a]) * padding;
            if(axisPadded != 0u && padded > static_cast<std::uint64_t>(kMaxPaddedNodes) / axisPadded)
                throw std::runtime_error(
                    "makeGrid: padded FFT grid for output grid " + std::to_string(grid.n[0]) + "x"
                    + std::to_string(grid.n[1]) + "x" + std::to_string(grid.n[2])
                    + " with padding " + std::to_string(padding)
                    + " is infeasible (limit " + std::to_string(kMaxPaddedNodes)
                    + " nodes); check the input gridSpacing/gridUnitSI or pass --cell-size");
            padded *= axisPadded;
        }
        return grid;
    }

    Field computeField(
        Bunch const& bunch,
        ComputeOptions const& options,
        StageTimes* stageTimes,
        VelocityBinning* outBinning)
    {
        // Both near-shell backends share the tiled assembly
        // F = F_mesh(all) - F_mesh(near) + F_exact(near); only the per-pair
        // evaluator differs (adaptive reference vs closed form).
        if(options.nearShell.mode != NearShellOptions::Mode::None)
            return computeFieldNearExact(bunch, options, options.nearShell, stageTimes, outBinning);

        // A multi-velocity bunch is the sum over velocity classes: the kernel
        // is linear in charge, so each velocity box contributes its own
        // deposit, kernel spectrum and convolution, accumulated into the same
        // output.  The box geometry depends only on the u-space extent, so a
        // fixed binning (the near shell passes the full-bunch one down) stays
        // bin-consistent on any sub-bunch and the near shell is bin-free.
        VelocityBinning derivedBinning;
        VelocityBinning const* binningPointer
            = options.fixedBinning ? &*options.fixedBinning : nullptr;
        if(!binningPointer)
        {
            derivedBinning = deriveVelocityBinning(
                bunch, options.tolerance, options.velocityBinning);
            binningPointer = &derivedBinning;
        }
        if(outBinning)
            *outBinning = *binningPointer;
        VelocityBinning const& binning = *binningPointer;
        std::size_t const binCount = binning.count;

        // Assignment of the actual bunch to the boxes.  A fixed binning was
        // derived for the full bunch, so a sub-bunch (the near shell tile
        // meshes) re-derives its own membership from the shared geometry.
        std::vector<std::size_t> const boxOf = [&] {
            if(!options.fixedBinning)
                return binning.boxOf;
            std::vector<std::size_t> assignment(bunch.size());
            for(std::size_t p = 0; p < bunch.size(); ++p)
            {
                double const coords[3] = {
                    bunch.ux[p] * binning.basis[0][0] + bunch.uy[p] * binning.basis[0][1]
                        + bunch.uz[p] * binning.basis[0][2],
                    bunch.ux[p] * binning.basis[1][0] + bunch.uy[p] * binning.basis[1][1]
                        + bunch.uz[p] * binning.basis[1][2],
                    bunch.ux[p] * binning.basis[2][0] + bunch.uy[p] * binning.basis[2][1]
                        + bunch.uz[p] * binning.basis[2][2]};
                assignment[p] = binning.boxIndex(coords[0], coords[1], coords[2]);
            }
            return assignment;
        }();
        std::vector<std::vector<std::size_t>> members(binCount);
        for(std::size_t p = 0; p < bunch.size(); ++p)
            members[boxOf[p]].push_back(p);

        Grid const grid = makeGrid(bunch, options);
        Field field;
        field.grid = grid;
        for(auto& component : field.data)
            component.assign(grid.size(), 0.0);

        StageTimes localTimes;
        StageTimes& times = stageTimes ? *stageTimes : localTimes;
        times = StageTimes{};

#if defined(PICBUNCH_BACKEND_CUDA)
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
#elif defined(PICBUNCH_BACKEND_HIP)
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
#else
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
#endif
        if(!selector.isAvailable())
            throw std::runtime_error("computeField: no device available for the compiled backend");
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        std::uint32_t const Mx = static_cast<std::uint32_t>(grid.n[0] * grid.fftPadding);
        std::uint32_t const My = static_cast<std::uint32_t>(grid.n[1] * grid.fftPadding);
        std::uint32_t const Mz = static_cast<std::uint32_t>(grid.n[2] * grid.fftPadding);
        // Mx may be odd: the R2C half-spectrum then has (Mx+1)/2 planes and no Nyquist
        // plane; KernelBuildKernel's parity-correct centred index handles that.  The
        // integer ``Mx/2 + 1`` already equals (Mx+1)/2 for odd Mx and Mx/2+1 for even.
        std::uint32_t const CP = Mx / 2u + 1u;
        // Flat layout is x fastest, then y, then z.  The plan's last dimension is x,
        // so the plan extents are (z, y, x) and the packed spectrum is (z, y, CP).
        Extents3 const realExt{Mz, My, Mx};
        Extents3 const specExt{Mz, My, CP};

        auto const cell = std::array<double, 3>{
            grid.spacing[0] / kRefinement,
            grid.spacing[1] / kRefinement,
            grid.spacing[2] / kRefinement};

        auto rhoReal = alpaka::onHost::allocUnified<double>(device, realExt);
        auto rhoSpec = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto kernelReal = alpaka::onHost::allocUnified<double>(device, realExt);
        auto kernelSpec = alpaka::onHost::allocUnified<Complex>(device, specExt);
        auto outReal = alpaka::onHost::allocUnified<double>(device, realExt);

        // Serial host TSC deposit of a particle list, in list order.  Same
        // stencil as the device kernel, without atomicAdd's unspecified order.
        auto depositHost = [&](std::vector<std::size_t> const& list) {
            double* rho = rhoReal.data();
            double const support = 1.5;
            double const org[3] = {grid.origin[0], grid.origin[1], grid.origin[2]};
            double const sp[3] = {grid.spacing[0], grid.spacing[1], grid.spacing[2]};
            for(std::size_t const p : list)
            {
                double const pos[3] = {bunch.x[p], bunch.y[p], bunch.z[p]};
                long lo[3];
                long hi[3];
                double t[3];
                for(int a = 0; a < 3; ++a)
                {
                    t[a] = (pos[a] - org[a]) / sp[a];
                    lo[a] = static_cast<long>(::ceil(t[a] - support));
                    hi[a] = static_cast<long>(::floor(t[a] + support));
                }
                double const q = bunch.weight[p] * bunch.charge[p];
                for(long ia = lo[0]; ia <= hi[0]; ++ia)
                {
                    if(ia < 0 || ia >= static_cast<long>(grid.n[0]))
                        continue;
                    double const wa = tscShape(static_cast<double>(ia) - t[0]);
                    if(wa == 0.0)
                        continue;
                    for(long ib = lo[1]; ib <= hi[1]; ++ib)
                    {
                        if(ib < 0 || ib >= static_cast<long>(grid.n[1]))
                            continue;
                        double const wb = wa * tscShape(static_cast<double>(ib) - t[1]);
                        if(wb == 0.0)
                            continue;
                        for(long ic = lo[2]; ic <= hi[2]; ++ic)
                        {
                            if(ic < 0 || ic >= static_cast<long>(grid.n[2]))
                                continue;
                            double const wc = wb * tscShape(static_cast<double>(ic) - t[2]);
                            if(wc == 0.0)
                                continue;
                            auto const idx = static_cast<std::size_t>(ia)
                                + static_cast<std::size_t>(Mx)
                                    * (static_cast<std::size_t>(ib)
                                       + static_cast<std::size_t>(My) * static_cast<std::size_t>(ic));
                            rho[idx] += wc * q;
                        }
                    }
                }
            }
        };

        // Device TSC deposit of a particle list into the (already zeroed) rho.
        auto depositDevice = [&](std::vector<std::size_t> const& list) {
            auto const numParticles = static_cast<std::uint32_t>(list.size());
            auto posX = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posY = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto posZ = alpaka::onHost::allocUnified<double>(device, numParticles);
            auto charge = alpaka::onHost::allocUnified<double>(device, numParticles);
            for(std::uint32_t m = 0; m < numParticles; ++m)
            {
                std::size_t const p = list[m];
                posX.data()[m] = bunch.x[p];
                posY.data()[m] = bunch.y[p];
                posZ.data()[m] = bunch.z[p];
                charge.data()[m] = bunch.weight[p] * bunch.charge[p];
            }
            std::uint32_t const frame = 64u;
            queue.enqueue(
                alpaka::onHost::FrameSpec{(numParticles + frame - 1u) / frame, frame},
                TscDepositKernel{},
                posX.data(),
                posY.data(),
                posZ.data(),
                charge.data(),
                numParticles,
                rhoReal.data(),
                Mx,
                My,
                static_cast<std::uint32_t>(grid.n[0]),
                static_cast<std::uint32_t>(grid.n[1]),
                static_cast<std::uint32_t>(grid.n[2]),
                grid.origin[0],
                grid.origin[1],
                grid.origin[2],
                grid.spacing[0],
                grid.spacing[1],
                grid.spacing[2]);
            // The host queue is non-blocking (a CallbackThread worker runs the
            // kernel), so the particle buffers must stay alive until the wait.
            // Keep this wait inside the block that owns posX/posY/posZ/charge.
            alpaka::onHost::wait(queue);
        };

        // ---------------------------------------------------------------- FFT
#if defined(ALPAKAV_DEP_ROCFFT) && ALPAKAV_DEP_ROCFFT
        // rocFFT keeps a process-global pool of HIP streams and destroys it from
        // a static destructor at library unload.  On ROCm 7.2.0 that destructor
        // calls hipStreamDestroy after the HIP runtime has already finalised and
        // segfaults.  Tear the pool down here, after the plans below are destroyed
        // but while the queue and HIP runtime are still alive.
        struct RocfftTeardownGuard
        {
            decltype(queue)& queue;
            ~RocfftTeardownGuard()
            {
                alpaka::onHost::wait(queue);
                static_cast<void>(rocfft_cleanup());
            }
        } rocfftTeardownGuard{queue};
#endif
        auto r2cPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.r2c().outOfPlace().build(device);
        auto c2rPlan
            = alpaka::fft::onHost::PlanBuilder<double, Extents3>{realExt}.c2r().outOfPlace().build(device);

        double const scale = (kRefinement * kRefinement * kRefinement)
            / (static_cast<double>(Mx) * static_cast<double>(My) * static_cast<double>(Mz));
        double const eFactor = kCoulombConstant;
        double const bFactor = kCoulombConstant / kSpeedOfLight;

        std::uint32_t const spectrumCount = Mz * My * CP;

        // Component/bin parallelism.  Every rank deposits and transforms the
        // same source spectrum, then owns a disjoint subset of the six output
        // components.  There is no distributed FFT: the transform stays whole
        // on each rank, and the six component arrays are summed afterwards.
        // A rank with no owned component still participates in the reduction.
        ParallelContext const* par = options.parallel;
        int const parSize = (par && par->distributed()) ? par->size : 1;
        int const parRank = (par && par->distributed()) ? par->rank : 0;

        // One velocity class: deposit, kernel spectrum and convolution,
        // accumulated into the shared output.  Empty boxes cost nothing.
        for(std::size_t b = 0; b < binCount; ++b)
        {
            if(members[b].empty())
                continue;
            if(members[b].size() > 0xffffffffu)
                throw std::runtime_error("computeField: particle count exceeds uint32 range");

            // -------------------------------------------------------- deposit
            auto const depositStart = Clock::now();
            alpaka::onHost::memset(queue, rhoReal, std::uint8_t{0});
            if(options.deterministicDeposit)
            {
                // The memset is queued on the device; finish it before the host
                // touches the same unified buffer.
                alpaka::onHost::wait(queue);
                depositHost(members[b]);
            }
            else
            {
                alpaka::onHost::wait(queue);
                depositDevice(members[b]);
            }
            times.depositMs += elapsedMs(depositStart);

            // ------------------------------------------------------------- FFT
            auto const fftStart = Clock::now();
            alpaka::fft::onHost::executeForward(queue, r2cPlan, rhoReal, rhoSpec);
            alpaka::onHost::wait(queue);
            times.fftMs += elapsedMs(fftStart);

            std::array<double, 3> const uRep{
                binning.representativeU[b][0],
                binning.representativeU[b][1],
                binning.representativeU[b][2]};

            for(int c = 0; c < 6; ++c)
            {
                if(parSize > 1 && (c % parSize) != parRank)
                    continue; // another rank owns this component
                // ------------------------------------------------------------ kernel
                auto const kernelStart = Clock::now();
                queue.enqueue(
                    alpaka::onHost::FrameSpec{
                        ceilDivVec(Extents3{Mx, My, Mz}, Extents3::fill(4u)),
                        Extents3::fill(4u)},
                    KernelBuildKernel{},
                    kernelReal.data(),
                    Mx,
                    My,
                    Mz,
                    c,
                    kYeeOffset[c][0],
                    kYeeOffset[c][1],
                    kYeeOffset[c][2],
                    cell[0],
                    cell[1],
                    cell[2],
                    kRefinement,
                    uRep[0],
                    uRep[1],
                    uRep[2]);
                alpaka::onHost::wait(queue);
                times.kernelMs += elapsedMs(kernelStart);

                // --------------------------------------------------------- spectrum
                auto const spectrumStart = Clock::now();
                alpaka::fft::onHost::executeForward(queue, r2cPlan, kernelReal, kernelSpec);
                alpaka::onHost::wait(queue);
                // product of two packed half-spectra; Complex<double> is two adjacent doubles
                static_assert(sizeof(Complex) == 2u * sizeof(double), "unexpected Complex layout");
                queue.enqueue(
                    alpaka::onHost::FrameSpec{(spectrumCount + 63u) / 64u, 64u},
                    MultiplySpectrumKernel{},
                    reinterpret_cast<double*>(kernelSpec.data()),
                    reinterpret_cast<double const*>(rhoSpec.data()),
                    spectrumCount);
                alpaka::onHost::wait(queue);
                alpaka::fft::onHost::executeBackward(queue, c2rPlan, kernelSpec, outReal);
                alpaka::onHost::wait(queue);
                times.fftMs += elapsedMs(spectrumStart);

                // ---------------------------------------------------------- read-out
                auto const readoutStart = Clock::now();
                double const factor = scale * (c < 3 ? eFactor : bFactor);
                for(std::uint32_t k = 0; k < static_cast<std::uint32_t>(grid.n[2]); ++k)
                    for(std::uint32_t j = 0; j < static_cast<std::uint32_t>(grid.n[1]); ++j)
                        for(std::uint32_t i = 0; i < static_cast<std::uint32_t>(grid.n[0]); ++i)
                        {
                            auto const in = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(Mx)
                                    * (static_cast<std::size_t>(j)
                                       + static_cast<std::size_t>(My) * static_cast<std::size_t>(k));
                            auto const out = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(grid.n[0])
                                    * (static_cast<std::size_t>(j)
                                       + static_cast<std::size_t>(grid.n[1]) * static_cast<std::size_t>(k));
                            // Accumulate: the field of the mixed bunch is the sum
                        // over the velocity classes.
                        field.data[c][out] += outReal.data()[in] * factor;
                        }
                times.readoutMs += elapsedMs(readoutStart);
            }
        }

        if(parSize > 1)
            for(int c = 0; c < 6; ++c)
                par->allreduceSum(field.data[c].data(), field.data[c].size());

        return field;
    }

} // namespace picBunch
