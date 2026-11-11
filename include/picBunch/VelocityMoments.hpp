// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Velocity-moment spectral decomposition (candidate C1).  The mesh kernel is
// exact only for a single shared ``u``; the binned path replaces every particle
// ``u`` by its box representative and pays one FFT chain per box.  Here the
// six-component point Lienard field is expanded in a multivariate Taylor series
// in velocity about the absolute-charge-weighted mean ``u0``:
//
//   K(u_p; R) ~= sum_{|a|<=n} (u_p - u0)^a  K_a(R)      (power form)
//
// where ``K_a(R)`` is the coefficient of ``(u-u0)^a`` (i.e. ``1/a!`` times the
// ``a``-th velocity derivative at ``u0``).  The field then becomes
//
//   F_c(x) = sum_{|a|<=n} conv( M_a, K_a^c )
//
// with the moment densities ``M_a(x) = sum_p q_p (u_p-u0)^a S(x-x_p)`` deposited
// once (a single multi-component pass) and one inverse transform per component.
//
// ``pointFieldNormalized`` is an elementary (real-analytic for R != 0) function
// of ``u``, so the derivatives are evaluated with a truncated multivariate
// Taylor automatic-differentiation (``TaylorPoly``) instead of by hand: the same
// code produces every derivative of every component and is unit-tested against
// finite differences.  Because ``B = beta x E`` with ``beta = u/gamma`` (both
// elementary in ``u``), the magnetic components fall out of the cross product of
// the ``E`` Taylor and the ``beta`` Taylor -- no separate B kernel algebra.

#pragma once

#include "picBunch/Types.hpp"

#include <cmath>
#include <cstddef>

// Use alpaka's host/device annotation from the single alpaka macro header.
// Include it unconditionally so ALPAKA_FN_ACC is defined for host and device
// translation units alike.
#include <alpaka/core/common.hpp>

namespace picBunch
{
    namespace detail
    {
        //! Number of multi-indices ``(a1,a2,a3)`` with ``a1+a2+a3 <= Order``.
        ALPAKA_FN_ACC inline constexpr std::size_t velocityMomentTerms(std::size_t order)
        {
            return (order + 1u) * (order + 2u) * (order + 3u) / 6u;
        }

        /*! Truncated multivariate Taylor polynomial in three variables, power form.
         *
         * ``c[compactIndex(a1,a2,a3)]`` is the coefficient of
         * ``(du1,du2,du3)^a`` of the value expanded about a centre, i.e. the
         * ``a``-th derivative divided by ``a!``.  Only the ``Order``-simplex
         * (``a1+a2+a3 <= Order``) is stored, packed by ``compactIndex``.
         *
         * The arithmetic (``tpMul``, ``tpInv``, ``tpSqrt``) is exact for the
         * truncated power series: division and the square root use the geometric
         * and binomial series, both of which converge to the correct
         * ``Order``-truncation whenever the series has a nonzero constant term
         * (which holds here: ``g2 = 1+u^2 >= 1`` and ``D > 0`` off the origin).
         */
        template<std::size_t Order>
        struct TaylorPoly
        {
            static constexpr std::size_t T = velocityMomentTerms(Order);
            double c[T]{};

            //! Compact (lexicographic in a3,a2,a1) index of a simplex multi-index.
            ALPAKA_FN_ACC inline static constexpr std::size_t compactIndex(std::size_t a1, std::size_t a2, std::size_t a3)
            {
                // The caller iterates the Order-simplex only; the result is valid
                // for ``a1+a2+a3 <= Order``.
                std::size_t const N = Order;
                std::size_t const k = a3;
                // Offset of the a3 = k block: sum_{k' = 0}^{k-1} C(N-k'),
                // C(M) = (M+1)(M+2)/2 is the number of (a1,a2) with a1+a2 <= M.
                std::size_t L = 0u;
                if(k > 0u)
                {
                    std::size_t const a = N - k + 1u;
                    std::size_t const b = N;
                    auto const s2 = [](std::size_t n) {
                        return n * (n + 1u) * (2u * n + 1u) / 6u;
                    };
                    auto const s1 = [](std::size_t n) {
                        return n * (n + 1u) / 2u;
                    };
                    L = (s2(b) - s2(a - 1u) + 3u * (s1(b) - s1(a - 1u)) + 2u * (b - a + 1u)) / 2u;
                }
                // Within the block, offset for a2: sum_{j' = 0}^{a2-1} (N-k-j'+1)
                // = C(N-k) - C(N-k-a2).
                std::size_t const w = N - k;
                std::size_t const Cw = (w + 1u) * (w + 2u) / 2u;
                std::size_t const Cwj = (w - a2 + 1u) * (w - a2 + 2u) / 2u;
                return L + (Cw - Cwj) + a1;
            }
        };

        template<std::size_t Order>
        ALPAKA_FN_ACC inline void tpClear(TaylorPoly<Order>& t)
        {
            for(std::size_t i = 0u; i < TaylorPoly<Order>::T; ++i)
                t.c[i] = 0.0;
        }

        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpScalar(double v)
        {
            TaylorPoly<Order> t;
            t.c[0] = v;
            return t;
        }

        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpAdd(TaylorPoly<Order> const& a, TaylorPoly<Order> const& b)
        {
            TaylorPoly<Order> r;
            for(std::size_t i = 0u; i < TaylorPoly<Order>::T; ++i)
                r.c[i] = a.c[i] + b.c[i];
            return r;
        }

        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpSub(TaylorPoly<Order> const& a, TaylorPoly<Order> const& b)
        {
            TaylorPoly<Order> r;
            for(std::size_t i = 0u; i < TaylorPoly<Order>::T; ++i)
                r.c[i] = a.c[i] - b.c[i];
            return r;
        }

        //! Truncated product: ``c[a] = sum_{i+j=a} A[i] B[j]`` over the simplex.
        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpMul(TaylorPoly<Order> const& a, TaylorPoly<Order> const& b)
        {
            using TP = TaylorPoly<Order>;
            TP r;
            for(std::size_t a3 = 0u; a3 <= Order; ++a3)
                for(std::size_t a2 = 0u; a2 <= Order - a3; ++a2)
                    for(std::size_t a1 = 0u; a1 <= Order - a3 - a2; ++a1)
                    {
                        double const av = a.c[TP::compactIndex(a1, a2, a3)];
                        if(av == 0.0)
                            continue;
                        std::size_t const rem = Order - (a1 + a2 + a3);
                        for(std::size_t b3 = 0u; b3 <= rem; ++b3)
                            for(std::size_t b2 = 0u; b2 <= rem - b3; ++b2)
                                for(std::size_t b1 = 0u; b1 <= rem - b2 - b3; ++b1)
                                {
                                    double const bv = b.c[TP::compactIndex(b1, b2, b3)];
                                    if(bv == 0.0)
                                        continue;
                                    r.c[TP::compactIndex(a1 + b1, a2 + b2, a3 + b3)] += av * bv;
                                }
                    }
            return r;
        }

        //! Truncated reciprocal ``1/b`` (requires ``b[0] != 0``).
        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpInv(TaylorPoly<Order> const& b)
        {
            using TP = TaylorPoly<Order>;
            double const b0 = b.c[0];
            double const invb0 = 1.0 / b0;
            TP w;
            for(std::size_t i = 0u; i < TP::T; ++i)
                w.c[i] = b.c[i] * invb0;
            w.c[0] = 0.0; // w = (b - b0) / b0 has zero constant term
            // 1/(1+w) = sum_{k=0}^{Order} (-w)^k  (exact for the Order-truncation)
            TP sum = tpScalar<Order>(1.0);
            TP power = tpScalar<Order>(1.0);
            for(std::size_t k = 1u; k <= Order; ++k)
            {
                power = tpMul<Order>(power, w);
                sum = (k % 2u == 1u) ? tpSub<Order>(sum, power) : tpAdd<Order>(sum, power);
            }
            TP r;
            for(std::size_t i = 0u; i < TP::T; ++i)
                r.c[i] = sum.c[i] * invb0;
            return r;
        }

        //! Truncated square root ``sqrt(b)`` (requires ``b[0] > 0``).
        template<std::size_t Order>
        ALPAKA_FN_ACC inline TaylorPoly<Order> tpSqrt(TaylorPoly<Order> const& b)
        {
            using TP = TaylorPoly<Order>;
            double const b0 = b.c[0];
            double const sqrtb0 = std::sqrt(b0);
            double const invb0 = 1.0 / b0;
            TP w;
            for(std::size_t i = 0u; i < TP::T; ++i)
                w.c[i] = b.c[i] * invb0;
            w.c[0] = 0.0;
            // sqrt(1+w) = sum_{k=0}^{Order} binom(1/2, k) w^k  (binomial series)
            TP sum = tpScalar<Order>(1.0); // k = 0 term
            TP power = tpScalar<Order>(1.0);
            double binom = 1.0;
            for(std::size_t k = 1u; k <= Order; ++k)
            {
                binom *= (0.5 - static_cast<double>(k - 1u)) / static_cast<double>(k);
                power = tpMul<Order>(power, w);
                TP term;
                for(std::size_t i = 0u; i < TP::T; ++i)
                    term.c[i] = power.c[i] * binom;
                sum = tpAdd<Order>(sum, term);
            }
            TP r;
            for(std::size_t i = 0u; i < TP::T; ++i)
                r.c[i] = sum.c[i] * sqrtb0;
            return r;
        }

        /*! Six-component point Lienard field expanded in velocity about ``u0``.
         *
         * Fills ``out`` (``6 * TaylorPoly<Order>::T`` doubles, layout
         * ``[component][compactIndex]`` with component order ``E_x, E_y, E_z,
         * B_x, B_y, B_z``) with the power-form Taylor coefficients of the
         * normalized field (the same normalization as ``pointFieldNormalized``).
         *
         * The expansion variable is ``du = u - u0``; the coefficient at
         * ``compactIndex(a)`` is ``K_a(R) = 1/a! d^a K / du^a |_{u0}``.  At
         * ``a = 0`` the coefficient is exactly the field value at ``u0``.
         */
        template<std::size_t Order>
        ALPAKA_FN_ACC inline void pointFieldTaylor(
            double Rx,
            double Ry,
            double Rz,
            double u0x,
            double u0y,
            double u0z,
            double* out)
        {
            using TP = TaylorPoly<Order>;
            // u = u0 + du (the symbolic increment lives in the basis indices).
            TP u1; u1.c[TP::compactIndex(1u, 0u, 0u)] = 1.0; u1.c[0] = u0x;
            TP u2; u2.c[TP::compactIndex(0u, 1u, 0u)] = 1.0; u2.c[0] = u0y;
            TP u3; u3.c[TP::compactIndex(0u, 0u, 1u)] = 1.0; u3.c[0] = u0z;

            // g2 = 1 + u1^2 + u2^2 + u3^2 (so gamma^2 = g2, invG2 = 1/g2).
            TP const g2 = tpAdd<Order>(
                tpScalar<Order>(1.0),
                tpAdd<Order>(
                    tpMul<Order>(u1, u1),
                    tpAdd<Order>(tpMul<Order>(u2, u2), tpMul<Order>(u3, u3))));
            TP const invG2 = tpInv<Order>(g2);
            // invGamma = 1/gamma = sqrt(1/g2).  beta_i = u_i * invGamma.
            TP const invGamma = tpSqrt<Order>(invG2);
            TP const beta1 = tpMul<Order>(u1, invGamma);
            TP const beta2 = tpMul<Order>(u2, invGamma);
            TP const beta3 = tpMul<Order>(u3, invGamma);

            double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
            // rb = R . beta
            TP const rb = tpAdd<Order>(
                tpAdd<Order>(
                    tpMul<Order>(beta1, tpScalar<Order>(Rx)),
                    tpMul<Order>(beta2, tpScalar<Order>(Ry))),
                tpMul<Order>(beta3, tpScalar<Order>(Rz)));
            // D = r2 * invG2 + rb^2  (> 0 for R != 0)
            TP const D = tpAdd<Order>(
                tpMul<Order>(invG2, tpScalar<Order>(r2)),
                tpMul<Order>(rb, rb));
            // f = invG2 * D^(-3/2) = invG2 * (1/D) * sqrt(1/D)
            TP const invD = tpInv<Order>(D);
            TP const sqrtInvD = tpSqrt<Order>(invD);
            TP const f = tpMul<Order>(tpMul<Order>(invG2, invD), sqrtInvD);
            // E = f * R
            TP const Ex = tpMul<Order>(f, tpScalar<Order>(Rx));
            TP const Ey = tpMul<Order>(f, tpScalar<Order>(Ry));
            TP const Ez = tpMul<Order>(f, tpScalar<Order>(Rz));
            // B = beta x E
            TP const Bx = tpSub<Order>(tpMul<Order>(beta2, Ez), tpMul<Order>(beta3, Ey));
            TP const By = tpSub<Order>(tpMul<Order>(beta3, Ex), tpMul<Order>(beta1, Ez));
            TP const Bz = tpSub<Order>(tpMul<Order>(beta1, Ey), tpMul<Order>(beta2, Ex));

            double* o = out;
            for(int i = 0; i < 6; ++i)
            {
                TP const* src = (i == 0) ? &Ex : (i == 1) ? &Ey : (i == 2) ? &Ez
                    : (i == 3) ? &Bx : (i == 4) ? &By : &Bz;
                for(std::size_t a = 0u; a < TP::T; ++a)
                    o[a] = src->c[a];
                o += TP::T;
            }
        }

        //! Reconstruct the field component ``c`` from a power-form Taylor:
        //! ``sum_a coeffs[c*T + a] * du1^a1 du2^a2 du3^a3``.  (Test helper.)
        template<std::size_t Order>
        ALPAKA_FN_ACC inline double pointFieldTaylorValue(
            double const* coeffs,
            int component,
            double du1,
            double du2,
            double du3)
        {
            using TP = TaylorPoly<Order>;
            double const* base = coeffs + static_cast<std::size_t>(component) * TP::T;
            double sum = 0.0;
            for(std::size_t a3 = 0u; a3 <= Order; ++a3)
                for(std::size_t a2 = 0u; a2 <= Order - a3; ++a2)
                    for(std::size_t a1 = 0u; a1 <= Order - a3 - a2; ++a1)
                    {
                        double term = base[TP::compactIndex(a1, a2, a3)];
                        for(std::size_t i = 0u; i < a1; ++i)
                            term *= du1;
                        for(std::size_t i = 0u; i < a2; ++i)
                            term *= du2;
                        for(std::size_t i = 0u; i < a3; ++i)
                            term *= du3;
                        sum += term;
                    }
            return sum;
        }
    } // namespace detail

    /*! Diagnostics of a velocity-moment run (reported by the CLI). */
    struct VelocityMomentsReport
    {
        bool used = false;   //!< true when the moment path produced the delivered field
        bool fellBack = false; //!< true when the moment policy fell back to the binned path
        int order = 0;       //!< moment order used (2 or 3), 0 when the path was not used
        bool feasible = true; //!< false when the moment path was infeasible (size/memory)
        /*! The probe exceeded the convergence budget but the explicit order was
         * forced through by VelocityMomentsOptions::allowUnconverged.  The
         * delivered field is then known to be unconverged.
         */
        bool allowUnconverged = false;
        /*! The moment field (or its k-space probe) contained a NaN or Inf and
         * the binned path was used instead.  This fallback is never overridden.
         */
        bool nonFinite = false;
        double probe = 0.0; //!< relative k-space remainder probe (order 2/3, auto)
        Vec3 u0{}; //!< absolute-charge-weighted expansion centre
        double maxSpread = 0.0; //!< largest |u_p - u0| (velocity-space radius)
    };

    /** Absolute-charge-weighted mean momentum -- the expansion centre ``u0``.
     *
     * ``|q_p|`` weighting keeps the centre well defined for counter-streaming
     * (opposite-charge) bunches, where the signed sum can approach zero; for a
     * same-sign bunch it is identical to the charge-weighted mean.
     */
    Vec3 velocityMomentsCentre(Bunch const& bunch);

    //! Largest ``|u_p - u0|`` over the particles (velocity-space radius).
    double velocityMomentsRadius(Bunch const& bunch, Vec3 const& u0);

} // namespace picBunch
