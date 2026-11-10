// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// CUDA instantiation of the fast near-field device backend.  The kernel is a
// direct port of include/picBunch/detail/FastNear.hpp and of the validated
// tools/closed-form/gpu_extract.cu extract kernel (picongpu branch
// experimental/closed-form, b3699dff).  All precision-critical work is double.

#include "picBunch/detail/FastNearDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace picBunch::pbfast
{
    namespace
    {
        // ---- device algebra, mirroring FastNear.hpp -------------------------

        ALPAKA_FN_ACC inline double dtsc(double x)
        {
            double const ax = x < 0.0 ? -x : x;
            if(ax < 0.5)
                return 0.75 - ax * ax;
            if(ax < 1.5)
            {
                double const t = 1.5 - ax;
                return 0.5 * t * t;
            }
            return 0.0;
        }

        ALPAKA_FN_ACC inline double dTfun(double A, double B, double u)
        {
            if(std::abs(u) < 1e-300)
                return A * std::asinh(B / A) + B * std::asinh(A / B);
            return A * std::asinh(B / std::sqrt(u * u + A * A))
                + B * std::asinh(A / std::sqrt(u * u + B * B))
                - u * std::atan(A * B / (u * std::sqrt(A * A + u * u + B * B)));
        }

        ALPAKA_FN_ACC inline double dboxField8T(double const p[3], double const lo[3], double const hi[3], int c)
        {
            double E = 0.0;
            if(c == 0)
            {
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ey[a], ez[b], hi[0] - p[0]) - dTfun(ey[a], ez[b], p[0] - lo[0]);
            }
            else if(c == 1)
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ez[2] = {hi[2] - p[2], p[2] - lo[2]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ex[a], ez[b], hi[1] - p[1]) - dTfun(ex[a], ez[b], p[1] - lo[1]);
            }
            else
            {
                double const ex[2] = {hi[0] - p[0], p[0] - lo[0]};
                double const ey[2] = {hi[1] - p[1], p[1] - lo[1]};
                for(int a = 0; a < 2; ++a)
                    for(int b = 0; b < 2; ++b)
                        E += dTfun(ex[a], ey[b], hi[2] - p[2]) - dTfun(ex[a], ey[b], p[2] - lo[2]);
            }
            return E;
        }

        ALPAKA_FN_ACC inline void dapplyContract(
            double const v[3],
            double const n[3],
            double k,
            double out[3])
        {
            double const vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            out[0] = vn * n[0] + k * (v[0] - vn * n[0]);
            out[1] = vn * n[1] + k * (v[1] - vn * n[1]);
            out[2] = vn * n[2] + k * (v[2] - vn * n[2]);
        }

        ALPAKA_FN_ACC inline double dboxEdgePrimitive(double l, double hh, double d)
        {
            constexpr double kPi = 3.141592653589793238462643383279502884;
            double const a = std::sqrt(hh * hh + d * d);
            double const r = std::sqrt(a * a + l * l);
            double const asinhTerm = d * std::asinh(l / a);
            double atanTerm;
            if(d == 0.0)
                atanTerm = (l > 0.0 ? 0.5 * kPi : (l < 0.0 ? -0.5 * kPi : 0.0)) * hh;
            else
                atanTerm = hh * std::atan(hh * l / (d * r));
            return asinhTerm + atanTerm;
        }

        ALPAKA_FN_ACC inline double dboxFaceIntegral(double const q[4][3], double const n[3])
        {
            constexpr double kPi = 3.141592653589793238462643383279502884;
            double const h = q[0][0] * n[0] + q[0][1] * n[1] + q[0][2] * n[2];
            double const hh = h < 0.0 ? -h : h;
            double I = 0.0;
            for(int i = 0; i < 4; ++i)
            {
                int const j = (i + 1) % 4;
                double const e[3]
                    = {q[j][0] - q[i][0], q[j][1] - q[i][1], q[j][2] - q[i][2]};
                double const L = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
                double const m[3] = {e[0] / L, e[1] / L, e[2] / L};
                double const n2[3] = {
                    m[1] * n[2] - m[2] * n[1],
                    m[2] * n[0] - m[0] * n[2],
                    m[0] * n[1] - m[1] * n[0]};
                double const d = q[i][0] * n2[0] + q[i][1] * n2[1] + q[i][2] * n2[2];
                double const la = q[i][0] * m[0] + q[i][1] * m[1] + q[i][2] * m[2];
                double const lb = q[j][0] * m[0] + q[j][1] * m[1] + q[j][2] * m[2];
                I += dboxEdgePrimitive(lb, hh, d) - dboxEdgePrimitive(la, hh, d);
            }
            double const foot[3] = {h * n[0], h * n[1], h * n[2]};
            double u[3] = {q[1][0] - q[0][0], q[1][1] - q[0][1], q[1][2] - q[0][2]};
            double const ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            u[0] /= ul;
            u[1] /= ul;
            u[2] /= ul;
            double const w[3] = {
                n[1] * u[2] - n[2] * u[1],
                n[2] * u[0] - n[0] * u[2],
                n[0] * u[1] - n[1] * u[0]};
            bool inside = false;
            int jj = 3;
            for(int i = 0; i < 4; ++i)
            {
                double const xi = (q[i][0] - foot[0]) * u[0] + (q[i][1] - foot[1]) * u[1]
                    + (q[i][2] - foot[2]) * u[2];
                double const yi = (q[i][0] - foot[0]) * w[0] + (q[i][1] - foot[1]) * w[1]
                    + (q[i][2] - foot[2]) * w[2];
                double const xj = (q[jj][0] - foot[0]) * u[0] + (q[jj][1] - foot[1]) * u[1]
                    + (q[jj][2] - foot[2]) * u[2];
                double const yj = (q[jj][0] - foot[0]) * w[0] + (q[jj][1] - foot[1]) * w[1]
                    + (q[jj][2] - foot[2]) * w[2];
                if(((yi > 0.0) != (yj > 0.0))
                   && (0.0 < (xj - xi) * (0.0 - yi) / (yj - yi) + xi))
                    inside = !inside;
                jj = i;
            }
            if(inside)
                I -= 2.0 * kPi * hh;
            return I;
        }

        ALPAKA_FN_ACC inline void dexactObliqueBox(
            double const R0[3],
            double const cells[3],
            double const u[3],
            double Eraw[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const gam = std::sqrt(1.0 + uu2);
            double const un = std::sqrt(uu2);
            double const n[3] = {u[0] / un, u[1] / un, u[2] / un};
            double const invg = 1.0 / gam;
            double c0[3];
            dapplyContract(R0, n, invg, c0);
            double const ev[3][3] = {
                {1.5 * cells[0], 0.0, 0.0},
                {0.0, 1.5 * cells[1], 0.0},
                {0.0, 0.0, 1.5 * cells[2]}};
            double e[3][3];
            for(int a = 0; a < 3; ++a)
                dapplyContract(ev[a], n, invg, e[a]);
            double V[2][2][2][3];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        for(int k = 0; k < 3; ++k)
                            V[a][b][c][k] = c0[k] + (a ? e[0][k] : -e[0][k])
                                + (b ? e[1][k] : -e[1][k]) + (c ? e[2][k] : -e[2][k]);
            int const FI[6][4][3] = {
                {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
            double S[3] = {0.0, 0.0, 0.0};
            for(int f = 0; f < 6; ++f)
            {
                double q[4][3];
                for(int v = 0; v < 4; ++v)
                    for(int k = 0; k < 3; ++k)
                        q[v][k] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]][k];
                double nn[3] = {
                    (q[1][1] - q[0][1]) * (q[2][2] - q[0][2])
                        - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]),
                    (q[1][2] - q[0][2]) * (q[2][0] - q[0][0])
                        - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]),
                    (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0])};
                double const nl = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                nn[0] /= nl;
                nn[1] /= nl;
                nn[2] /= nl;
                double cen[3] = {0.0, 0.0, 0.0};
                for(int v = 0; v < 4; ++v)
                    for(int k = 0; k < 3; ++k)
                        cen[k] += 0.25 * q[v][k];
                if((cen[0] - c0[0]) * nn[0] + (cen[1] - c0[1]) * nn[1]
                       + (cen[2] - c0[2]) * nn[2]
                   < 0.0)
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        double const t = q[0][k];
                        q[0][k] = q[3][k];
                        q[3][k] = t;
                    }
                    for(int k = 0; k < 3; ++k)
                    {
                        double const t = q[1][k];
                        q[1][k] = q[2][k];
                        q[2][k] = t;
                    }
                    nn[0] = (q[1][1] - q[0][1]) * (q[2][2] - q[0][2])
                        - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]);
                    nn[1] = (q[1][2] - q[0][2]) * (q[2][0] - q[0][0])
                        - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]);
                    nn[2] = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]);
                    double const l2 = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                    nn[0] /= l2;
                    nn[1] /= l2;
                    nn[2] /= l2;
                }
                double const I = dboxFaceIntegral(q, nn);
                S[0] += I * nn[0];
                S[1] += I * nn[1];
                S[2] += I * nn[2];
            }
            double const F[3] = {-S[0], -S[1], -S[2]};
            dapplyContract(F, n, gam, Eraw); // M^{-1}
        }


        // ---- exact s-space evaluator, mirroring FastNear.hpp ----------------
        // Exact in-support TSC field in the source frame.  The host shares the
        // 108 unique face moments across neighbouring cells; here each of the
        // 27 cells evaluates its two faces per axis inline so no per-thread
        // 7.8 KB moment array is spilled to local memory.

        struct dSAxis
        {
            int a, b, c;
            double Hxx, Hxy, Hyy, detH;
            double L11, L21, L22, detL;
            double ax, bx, by;
            double uhx, uhy, qc;
        };

        ALPAKA_FN_ACC inline void dsPrepAxis(dSAxis& X, int a, double const n[3], double s)
        {
            int const b = (a + 1) % 3;
            int const c = (a + 2) % 3;
            double G[3][3];
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    G[i][j] = (i == j ? 1.0 : 0.0) + s * n[i] * n[j];
            X.a = a;
            X.b = b;
            X.c = c;
            X.Hxx = G[b][b];
            X.Hxy = G[b][c];
            X.Hyy = G[c][c];
            X.detH = X.Hxx * X.Hyy - X.Hxy * X.Hxy;
            double const gab = G[a][b], gac = G[a][c];
            X.uhx = -(X.Hyy * gab - X.Hxy * gac) / X.detH;
            X.uhy = -(-X.Hxy * gab + X.Hxx * gac) / X.detH;
            X.qc = G[a][a] + gab * X.uhx + gac * X.uhy;
            X.L11 = std::sqrt(X.Hxx);
            X.L21 = X.Hxy / X.L11;
            X.L22 = std::sqrt(X.Hyy - X.L21 * X.L21);
            X.detL = X.L11 * X.L22;
            X.ax = 1.0 / X.L11;
            X.bx = -X.L21 / (X.L11 * X.L22);
            X.by = 1.0 / X.L22;
        }

        ALPAKA_FN_ACC inline void dsEdgeW4(
            double xi0,
            double eta0,
            double dxi,
            double deta,
            double Q,
            double A,
            double invA,
            double sqA,
            double W[4])
        {
            double const inv2A = 0.5 * invA;
            double const B = 2.0 * (xi0 * dxi + eta0 * deta);
            double const C = Q + xi0 * xi0 + eta0 * eta0;
            double const t0 = -B * inv2A;
            double const Dd = xi0 * deta - eta0 * dxi;
            double s2 = (A * Q + Dd * Dd) / (A * A);
            if(s2 < 0.0)
                s2 = 0.0;
            double const u0 = -t0;
            double const u1 = 1.0 - t0;
            double const xi1 = xi0 + dxi;
            double const eta1 = eta0 + deta;
            double const rho0 = std::sqrt(C);
            double const rho1 = std::sqrt(Q + xi1 * xi1 + eta1 * eta1);
            double I0;
            if(s2 > 0.0)
            {
                double const ss = std::sqrt(s2);
                I0 = 0.5 * (u1 * rho1 - u0 * rho0)
                    + 0.5 * s2 * sqA * (std::asinh(u1 / ss) - std::asinh(u0 / ss));
            }
            else
                I0 = 0.5 * (u1 * rho1 - u0 * rho0);
            double const r1c = rho1 * rho1 * rho1;
            W[0] = I0;
            W[1] = (r1c - rho0 * rho0 * rho0) * (invA / 3.0) - (B * inv2A) * W[0];
            for(int k = 2; k <= 3; ++k)
            {
                double const ik = invA / (k + 2.0);
                W[k] = r1c * ik
                    - ((2.0 * k + 1.0) * B * inv2A / (k + 2.0)) * W[k - 1]
                    - ((k - 1.0) * C) * ik * W[k - 2];
            }
        }

        ALPAKA_FN_ACC inline void dsFaceFab4(double const q[4][3], double h, double Fab[3][5])
        {
            constexpr double kPi = 3.141592653589793238462643383279502884;
            for(int a = 0; a <= 2; ++a)
                for(int b = 0; b <= 4; ++b)
                    Fab[a][b] = 0.0;
            double const Q = h * h;
            double const foot[3] = {0.0, 0.0, h};
            double u[3] = {q[1][0] - q[0][0], q[1][1] - q[0][1], q[1][2] - q[0][2]};
            double const ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            u[0] /= ul;
            u[1] /= ul;
            u[2] /= ul;
            // n x u with n = (0,0,1): (-u_y, u_x, 0)
            double const vv[3] = {-u[1], u[0], 0.0};
            double xi[4], eta[4];
            for(int l = 0; l < 4; ++l)
            {
                double const w[3] = {q[l][0] - foot[0], q[l][1] - foot[1], q[l][2] - foot[2]};
                xi[l] = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
                eta[l] = w[0] * vv[0] + w[1] * vv[1] + w[2] * vv[2];
            }
            double A[4], invA[4];
            for(int l = 0; l < 4; ++l)
            {
                int const m = (l + 1) % 4;
                double const dx = xi[m] - xi[l], dy = eta[m] - eta[l];
                A[l] = dx * dx + dy * dy;
                invA[l] = 1.0 / A[l];
            }
            double sqA[4];
            sqA[0] = std::sqrt(A[0]);
            sqA[1] = std::sqrt(A[1]);
            sqA[2] = sqA[0];
            sqA[3] = sqA[1];
            {
                double const hh = std::abs(h);
                double Ibase = 0.0;
                for(int l = 0; l < 4; ++l)
                {
                    int const m = (l + 1) % 4;
                    double const ex = xi[m] - xi[l], ey = eta[m] - eta[l];
                    double const invL = 1.0 / std::sqrt(ex * ex + ey * ey);
                    double const mx = ex * invL, my = ey * invL;
                    double const d = xi[l] * my - eta[l] * mx;
                    double const la = xi[l] * mx + eta[l] * my;
                    double const lb = la + 1.0 / invL;
                    Ibase += dboxEdgePrimitive(lb, hh, d) - dboxEdgePrimitive(la, hh, d);
                }
                bool inside = false;
                int j = 3;
                for(int i = 0; i < 4; ++i)
                {
                    double const xiI = xi[i], yi = eta[i];
                    double const xj = xi[j], yj = eta[j];
                    if(((yi > 0.0) != (yj > 0.0))
                       && (0.0 < (xj - xiI) * (0.0 - yi) / (yj - yi) + xiI))
                        inside = !inside;
                    j = i;
                }
                if(inside)
                    Ibase -= 2.0 * kPi * hh;
                Fab[0][0] = Ibase;
            }
            double U1[4] = {0.0, 0.0, 0.0, 0.0};
            double U2[3] = {0.0, 0.0, 0.0};
            double V0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for(int l = 0; l < 4; ++l)
            {
                int const m = (l + 1) % 4;
                double const xa = xi[l], ya = eta[l];
                double const dx = xi[m] - xi[l], dy = eta[m] - eta[l];
                double W[4];
                dsEdgeW4(xa, ya, dx, dy, Q, A[l], invA[l], sqA[l], W);
                double PY[4][4] = {};
                PY[0][0] = 1.0;
                for(int nn = 1; nn <= 3; ++nn)
                    for(int k = 0; k <= nn; ++k)
                        PY[nn][k] = (k <= nn - 1 ? PY[nn - 1][k] * ya : 0.0)
                            + (k >= 1 ? PY[nn - 1][k - 1] * dy : 0.0);
                double MY[4], MW[3];
                for(int nn = 0; nn <= 3; ++nn)
                {
                    double sum = 0.0;
                    for(int j = 0; j <= nn; ++j)
                        sum += PY[nn][j] * W[j];
                    MY[nn] = sum;
                }
                for(int nn = 0; nn <= 2; ++nn)
                {
                    double sum = 0.0;
                    for(int j = 0; j <= nn; ++j)
                        sum += PY[nn][j] * W[j + 1];
                    MW[nn] = sum;
                }
                for(int b = 0; b <= 3; ++b)
                    U1[b] += dy * MY[b];
                for(int b = 0; b <= 2; ++b)
                    U2[b] += dy * (xa * MY[b] + dx * MW[b]);
                for(int b = 1; b <= 4; ++b)
                    V0[b] += -dx * MY[b - 1];
            }
            for(int deg = 1; deg <= 4; ++deg)
            {
                if(deg == 1)
                    Fab[0][1] = V0[1];
                else
                    Fab[0][deg] = (2.0 * V0[deg]
                                   - (deg - 1.0) * (Q * Fab[0][deg - 2] + U2[deg - 2]))
                        / (deg + 1.0);
                Fab[1][deg - 1] = U1[deg - 1];
                if(deg >= 2)
                    Fab[2][deg - 2]
                        = (U2[deg - 2] - (Q * Fab[0][deg - 2] + Fab[0][deg])) / 2.0;
            }
        }

        ALPAKA_FN_ACC inline void dsFaceM(
            dSAxis const& X,
            double h,
            double const Rc[3],
            double const C[3],
            double gam,
            double M[3][3])
        {
            double const qmin = h * h * X.qc;
            double const hh = std::sqrt(qmin > 0.0 ? qmin : 0.0);
            double const u0x = h * X.uhx, u0y = h * X.uhy;
            int const b = X.b, c = X.c;
            double const xlo = Rc[b] - 0.5 * C[b], xhi = Rc[b] + 0.5 * C[b];
            double const ylo = Rc[c] - 0.5 * C[c], yhi = Rc[c] + 0.5 * C[c];
            double const xy[4][2] = {{xlo, ylo}, {xhi, ylo}, {xhi, yhi}, {xlo, yhi}};
            double qz[4][3];
            for(int i = 0; i < 4; ++i)
            {
                double const dx = xy[i][0] - u0x, dy = xy[i][1] - u0y;
                qz[i][0] = X.L11 * dx + X.L21 * dy;
                qz[i][1] = X.L22 * dy;
                qz[i][2] = hh;
            }
            double Fab[3][5];
            dsFaceFab4(qz, hh, Fab);
            double const ax = X.ax, bx = X.bx, by = X.by;
            double const ux[3] = {1.0, u0x, u0x * u0x};
            double const uy[3] = {1.0, u0y, u0y * u0y};
            double const Ax[3] = {1.0, ax, ax * ax};
            double const Bx[3] = {1.0, bx, bx * bx};
            double const By[3] = {1.0, by, by * by};
            static int const CB[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
            for(int p = 0; p <= 2; ++p)
                for(int q = 0; q <= 2; ++q)
                {
                    double sum = 0.0;
                    for(int i = 0; i <= p; ++i)
                    {
                        double const ci = CB[p][i] * ux[p - i];
                        for(int k = 0; k <= i; ++k)
                        {
                            double const ck = CB[i][k] * Ax[i - k] * Bx[k];
                            for(int j = 0; j <= q; ++j)
                                sum += ci * ck * (CB[q][j] * uy[q - j] * By[j])
                                    * Fab[i - k][k + j];
                        }
                    }
                    M[p][q] = gam / X.detL * sum;
                }
        }

        //! Knot-plane snap helpers; mirror FastNear.hpp.
        inline constexpr double dkKnotDetect = 1.0e-6;
        inline constexpr double dkKnotEvalSnapLowGamma = 1.0e-9;
        inline constexpr double dkKnotEvalSnapHighGamma = 1.0e-7;
        inline constexpr double dkKnotHighGammaThreshold = 20.0;
        inline constexpr double dkKnotGateSnap = 1.0e-5;

        ALPAKA_FN_ACC inline double dknotEvalSnap(double const gamma)
        {
            return gamma >= dkKnotHighGammaThreshold ? dkKnotEvalSnapHighGamma
                                                     : dkKnotEvalSnapLowGamma;
        }

        ALPAKA_FN_ACC inline double dsnapKnotCoordinate(double const s, double const snap)
        {
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};
            for(int k = 0; k < 4; ++k)
                if(std::abs(s - knots[k]) < dkKnotDetect)
                    return knots[k] + (knots[k] > 0.0 ? -snap : snap);
            return s;
        }

        ALPAKA_FN_ACC inline void dsnapKnotOrigin(
            double const R0[3],
            double const cells[3],
            double const snap,
            double out[3])
        {
            for(int a = 0; a < 3; ++a)
                out[a] = cells[a] * dsnapKnotCoordinate(R0[a] / cells[a], snap);
        }

        ALPAKA_FN_ACC inline void dpolyBoxFieldS(
            double const R0[3],
            double const cells[3],
            double const u[3],
            double E[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const gam = std::sqrt(1.0 + uu2);
            double n[3] = {0.0, 0.0, 0.0};
            if(uu2 > 0.0)
            {
                double const un = std::sqrt(uu2);
                n[0] = u[0] / un;
                n[1] = u[1] / un;
                n[2] = u[2] / un;
            }
            double const s = gam * gam - 1.0;
            dSAxis AX[3];
            for(int a = 0; a < 3; ++a)
                dsPrepAxis(AX[a], a, n, s);
            double const qA[3] = {0.5, -1.0, 0.5};
            double const qB[3] = {1.5, 0.0, -1.5};
            double const qC[3] = {1.125, 0.75, 1.125};
            double rr[3];
            dsnapKnotOrigin(R0, cells, dknotEvalSnap(gam), rr);
            double Iv[3] = {0.0, 0.0, 0.0};
            for(int sx = -1; sx <= 1; ++sx)
                for(int sy = -1; sy <= 1; ++sy)
                    for(int sz = -1; sz <= 1; ++sz)
                    {
                        int const sc[3] = {sx, sy, sz};
                        double pc[3][3];
                        for(int a = 0; a < 3; ++a)
                        {
                            int const k = sc[a] + 1;
                            double const ic = 1.0 / cells[a], t = rr[a] * ic;
                            pc[a][0] = qA[k] * t * t + qB[k] * t + qC[k];
                            pc[a][1] = -2.0 * qA[k] * t * ic - qB[k] * ic;
                            pc[a][2] = qA[k] * ic * ic;
                        }
                        double pB[27];
                        for(int i = 0; i < 3; ++i)
                            for(int j = 0; j < 3; ++j)
                                for(int k = 0; k < 3; ++k)
                                    pB[i * 9 + j * 3 + k] = pc[0][i] * pc[1][j] * pc[2][k];
                        // Per-cell face moments (both planes per axis), then
                        // the same Wt accumulation as the host evaluator.
                        double Mface[3][2][3][3];
                        double hface[3][2];
                        for(int a = 0; a < 3; ++a)
                        {
                            int const b = (a + 1) % 3, c = (a + 2) % 3;
                            double Rc[3] = {0.0, 0.0, 0.0};
                            Rc[b] = rr[b] - cells[b] * static_cast<double>(sc[b]);
                            Rc[c] = rr[c] - cells[c] * static_cast<double>(sc[c]);
                            for(int sg = 0; sg < 2; ++sg)
                            {
                                int const pl = (sg == 0) ? (sc[a] + 2) : (sc[a] + 1);
                                double const sp = -1.5 + static_cast<double>(pl);
                                double const h = rr[a] - cells[a] * sp;
                                hface[a][sg] = h;
                                dsFaceM(AX[a], h, Rc, cells, gam, Mface[a][sg]);
                            }
                        }
                        double Wt[27] = {};
                        for(int a = 0; a < 3; ++a)
                        {
                            int const b = (a + 1) % 3, c = (a + 2) % 3;
                            for(int sg = 0; sg < 2; ++sg)
                            {
                                double const h = hface[a][sg];
                                double const hp[3] = {h, h * h, h * h * h};
                                double const sign = sg ? 1.0 : -1.0;
                                for(int i = 0; i < 3; ++i)
                                    for(int j = 0; j < 3; ++j)
                                        for(int k = 0; k < 3; ++k)
                                        {
                                            int const be[3] = {i, j, k};
                                            Wt[i * 9 + j * 3 + k]
                                                += sign * hp[be[a]] * Mface[a][sg][be[b]][be[c]];
                                        }
                            }
                        }
                        for(int a = 0; a < 3; ++a)
                        {
                            double acc = 0.0;
                            for(int i = 0; i < 3; ++i)
                                for(int j = 0; j < 3; ++j)
                                    for(int k = 0; k < 3; ++k)
                                    {
                                        int const be[3] = {i, j, k};
                                        double const pv = pB[i * 9 + j * 3 + k];
                                        if(be[a] == 0 || pv == 0.0)
                                            continue;
                                        int g[3] = {i, j, k};
                                        g[a]--;
                                        int const dg = g[0] + g[1] + g[2];
                                        double const W
                                            = Wt[g[0] * 9 + g[1] * 3 + g[2]] / (dg + 2.0);
                                        acc += be[a] * pv * W;
                                    }
                            Iv[a] += acc;
                        }
                    }
            double const vn = Iv[0] * n[0] + Iv[1] * n[1] + Iv[2] * n[2];
            double const invV = 1.0 / (cells[0] * cells[1] * cells[2]);
            for(int a = 0; a < 3; ++a)
                E[a] = ((Iv[a] - vn * n[a]) + vn * n[a] / (gam * gam)) * invV;
        }


        //! Port of detail::polyBoxUsable's d-space conditioning check.
        ALPAKA_FN_ACC inline bool dPolyBoxUsable(
            double const R0[3],
            double const cells[3],
            double const u[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            if(!(uu2 > 1e-24) || !std::isfinite(uu2))
                return false;
            double const s[3] = {R0[0] / cells[0], R0[1] / cells[1], R0[2] / cells[2]};
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};
            double sc[3];
            for(int a = 0; a < 3; ++a)
            {
                if(!(s[a] > -1.5 && s[a] < 1.5))
                    return false;
                int const k = s[a] < -0.5 ? 0 : (s[a] < 0.5 ? 1 : 2);
                double const m = std::min(s[a] - knots[k], knots[k + 1] - s[a]);
                if(!(m > 1.0e-6))
                    return false;
                sc[a] = static_cast<double>(k - 1);
            }
            double const gam = std::sqrt(1.0 + uu2);
            double const un = std::sqrt(uu2);
            double const n[3] = {u[0] / un, u[1] / un, u[2] / un};
            double diff[3] = {
                R0[0] - sc[0] * cells[0],
                R0[1] - sc[1] * cells[1],
                R0[2] - sc[2] * cells[2]};
            double dc[3];
            dapplyContract(diff, n, 1.0 / gam, dc);
            double e[3][3];
            double ev[3][3] = {
                {cells[0], 0.0, 0.0},
                {0.0, cells[1], 0.0},
                {0.0, 0.0, cells[2]}};
            double emax = 0.0;
            for(int a = 0; a < 3; ++a)
            {
                dapplyContract(ev[a], n, 1.0 / gam, e[a]);
                double const l = std::sqrt(e[a][0] * e[a][0] + e[a][1] * e[a][1]
                                           + e[a][2] * e[a][2]);
                if(!(l > 0.0) || !std::isfinite(l))
                    return false;
                emax = std::max(emax, l);
            }
            if(!(std::isfinite(dc[0]) && std::isfinite(dc[1]) && std::isfinite(dc[2])))
                return false;
            double V[2][2][2][3];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        for(int k = 0; k < 3; ++k)
                            V[a][b][c][k] = dc[k] + (a ? 0.5 : -0.5) * e[0][k]
                                + (b ? 0.5 : -0.5) * e[1][k]
                                + (c ? 0.5 : -0.5) * e[2][k];
            int const FI[6][4][3] = {
                {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
            double hmin = 1e300;
            for(int f = 0; f < 6; ++f)
            {
                double q[4][3];
                for(int v = 0; v < 4; ++v)
                    for(int k = 0; k < 3; ++k)
                        q[v][k] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]][k];
                double nn[3] = {
                    (q[1][1] - q[0][1]) * (q[2][2] - q[0][2])
                        - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]),
                    (q[1][2] - q[0][2]) * (q[2][0] - q[0][0])
                        - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]),
                    (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0])};
                double const nl = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                if(!(nl > 0.0) || !std::isfinite(nl))
                    return false;
                nn[0] /= nl;
                nn[1] /= nl;
                nn[2] /= nl;
                double const h = q[0][0] * nn[0] + q[0][1] * nn[1] + q[0][2] * nn[2];
                if(!std::isfinite(h))
                    return false;
                hmin = std::min(hmin, std::abs(h));
            }
            return hmin > 1.0e-8 * emax;
        }

        //! Production validity gate (mirrors FastNear.hpp::sSpaceExactUsable).
        ALPAKA_FN_ACC inline bool dsSpaceExactUsable(
            double const R0[3],
            double const cells[3],
            double const u[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            if(!(uu2 >= 0.0) || !std::isfinite(uu2))
                return false;
            for(int a = 0; a < 3; ++a)
            {
                if(!(cells[a] > 0.0) || !std::isfinite(cells[a]) || !std::isfinite(R0[a]))
                    return false;
            }
            double Rg[3];
            dsnapKnotOrigin(R0, cells, dkKnotGateSnap, Rg);
            double const s[3] = {Rg[0] / cells[0], Rg[1] / cells[1], Rg[2] / cells[2]};
            bool inside = true;
            for(int a = 0; a < 3; ++a)
            {
                if(!std::isfinite(s[a]))
                    return false;
                inside = inside && std::abs(s[a]) < 1.5;
            }
            if(inside && uu2 > 0.0)
                return dPolyBoxUsable(Rg, cells, u);
            return true;
        }

        //! Source-relative TSC cell class; mirrors supportCellClass.
        ALPAKA_FN_ACC inline int dsupportCellClass(double const R0[3], double const cells[3])
        {
            int nonzero = 0;
            for(int a = 0; a < 3; ++a)
            {
                double const s = R0[a] / cells[a];
                int m = (std::abs(s) <= 0.5 + dkKnotDetect)
                    ? 0
                    : (s > 0.0 ? 1 : -1);
                if(m != 0)
                    ++nonzero;
            }
            return nonzero;
        }

        //! Exact-routing predicate; mirrors detail::routesExact.
        ALPAKA_FN_ACC inline bool droutesExact(
            double const R0[3],
            double const cells[3],
            double const u[3],
            DeviceParams const& prm)
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            bool inside = true;
            for(int a = 0; a < 3; ++a)
            {
                double const s = R0[a] / cells[a];
                if(!std::isfinite(s))
                    return false;
                inside = inside && std::abs(s) < 1.5;
            }
            if(inside)
            {
                if(prm.exactCellReach < 0)
                    return false;
                int const cls = dsupportCellClass(R0, cells);
                bool const selected = prm.exactCellReach >= 2
                    || (prm.exactCellReach == 1 && cls <= 1)
                    || (prm.exactCellReach == 0 && cls == 0);
                if(!selected)
                    return false;
                return dsSpaceExactUsable(R0, cells, u);
            }
            double const gam = std::sqrt(1.0 + uu2);
            int const need = quadratureOrderFor(gam, prm.tolerance, prm.order);
            if(!(gam >= kOutOfSupportExactGamma || need > kQuadratureOrderCap))
                return false;
            return dsSpaceExactUsable(R0, cells, u);
        }

        ALPAKA_FN_ACC inline void dsupportPlanes(
            double const R0[3],
            double const cells[3],
            double const u[3],
            double planesN[6][3],
            double planesH[6])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const gam = std::sqrt(1.0 + uu2);
            double const un = std::sqrt(uu2);
            double const n[3] = {u[0] / un, u[1] / un, u[2] / un};
            double const invg = 1.0 / gam;
            double c0[3];
            dapplyContract(R0, n, invg, c0);
            double const ev[3][3] = {
                {1.5 * cells[0], 0.0, 0.0},
                {0.0, 1.5 * cells[1], 0.0},
                {0.0, 0.0, 1.5 * cells[2]}};
            double e[3][3];
            for(int a = 0; a < 3; ++a)
                dapplyContract(ev[a], n, invg, e[a]);
            double V[2][2][2][3];
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int c = 0; c < 2; ++c)
                        for(int k = 0; k < 3; ++k)
                            V[a][b][c][k] = c0[k] + (a ? e[0][k] : -e[0][k])
                                + (b ? e[1][k] : -e[1][k]) + (c ? e[2][k] : -e[2][k]);
            int const FI[6][4][3] = {
                {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
            for(int f = 0; f < 6; ++f)
            {
                double q[4][3];
                for(int v = 0; v < 4; ++v)
                    for(int k = 0; k < 3; ++k)
                        q[v][k] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]][k];
                double nn[3] = {
                    (q[1][1] - q[0][1]) * (q[2][2] - q[0][2])
                        - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]),
                    (q[1][2] - q[0][2]) * (q[2][0] - q[0][0])
                        - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]),
                    (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0])};
                double const nl = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                nn[0] /= nl;
                nn[1] /= nl;
                nn[2] /= nl;
                double cen[3] = {0.0, 0.0, 0.0};
                for(int v = 0; v < 4; ++v)
                    for(int k = 0; k < 3; ++k)
                        cen[k] += 0.25 * q[v][k];
                if((cen[0] - c0[0]) * nn[0] + (cen[1] - c0[1]) * nn[1]
                       + (cen[2] - c0[2]) * nn[2]
                   < 0.0)
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        double const t = q[0][k];
                        q[0][k] = q[3][k];
                        q[3][k] = t;
                    }
                    for(int k = 0; k < 3; ++k)
                    {
                        double const t = q[1][k];
                        q[1][k] = q[2][k];
                        q[2][k] = t;
                    }
                    nn[0] = (q[1][1] - q[0][1]) * (q[2][2] - q[0][2])
                        - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]);
                    nn[1] = (q[1][2] - q[0][2]) * (q[2][0] - q[0][0])
                        - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]);
                    nn[2] = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]);
                    double const l2 = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                    nn[0] /= l2;
                    nn[1] /= l2;
                    nn[2] /= l2;
                }
                planesN[f][0] = nn[0];
                planesN[f][1] = nn[1];
                planesN[f][2] = nn[2];
                planesH[f] = q[0][0] * nn[0] + q[0][1] * nn[1] + q[0][2] * nn[2];
            }
        }

        ALPAKA_FN_ACC inline void dobliObliqueRayField(
            double const R0[3],
            double const cells[3],
            double const u[3],
            double Erem[3])
        {
            constexpr double kPi = 3.141592653589793238462643383279502884;
            constexpr double kMuX[16] = {-0.98940093499164994, -0.9445750230732326, -0.86563120238783176, -0.755404408355003, -0.61787624440264377, -0.45801677765722737, -0.28160355077925892, -0.095012509837637441, 0.095012509837637441, 0.28160355077925892, 0.45801677765722737, 0.61787624440264377, 0.755404408355003, 0.86563120238783176, 0.9445750230732326, 0.98940093499164994};
            constexpr double kMuW[16] = {0.027152459411754176, 0.062253523938647456, 0.095158511682492605, 0.12462897125553407, 0.14959598881657671, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.14959598881657671, 0.12462897125553407, 0.095158511682492605, 0.062253523938647456, 0.027152459411754176};
            constexpr double kPhX[32] = {-0.99726386184948157, -0.98561151154526838, -0.96476225558750639, -0.93490607593773967, -0.89632115576605209, -0.84936761373256997, -0.79448379596794239, -0.73218211874028971, -0.66304426693021523, -0.5877157572407623, -0.50689990893222936, -0.42135127613063533, -0.33186860228212767, -0.23928736225213706, -0.14447196158279649, -0.048307665687738324, 0.048307665687738324, 0.14447196158279649, 0.23928736225213706, 0.33186860228212767, 0.42135127613063533, 0.50689990893222936, 0.5877157572407623, 0.66304426693021523, 0.73218211874028971, 0.79448379596794239, 0.84936761373256997, 0.89632115576605209, 0.93490607593773967, 0.96476225558750639, 0.98561151154526838, 0.99726386184948157};
            constexpr double kPhW[32] = {0.0070186100094705058, 0.016274394730905743, 0.025392065309262024, 0.034273862913021765, 0.042835898022226836, 0.050998059262376091, 0.058684093478535565, 0.065822222776361683, 0.072345794108848338, 0.078193895787070228, 0.083311924226946707, 0.087652093004403783, 0.09117387869576378, 0.093844399080804511, 0.095638720079274708, 0.096540088514727659, 0.096540088514727659, 0.095638720079274708, 0.093844399080804511, 0.09117387869576378, 0.087652093004403783, 0.083311924226946707, 0.078193895787070228, 0.072345794108848338, 0.065822222776361683, 0.058684093478535565, 0.050998059262376091, 0.042835898022226836, 0.034273862913021765, 0.025392065309262024, 0.016274394730905743, 0.0070186100094705058};
            constexpr double kRadX[4] = {
                -0.86113631159405258, -0.33998104358485626, 0.33998104358485626,
                0.86113631159405258};
            constexpr double kRadW[4] = {
                0.34785484513745386, 0.65214515486254609, 0.65214515486254609,
                0.34785484513745386};
            constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};

            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const gam = std::sqrt(1.0 + uu2);
            double const un = std::sqrt(uu2);
            double const n[3] = {u[0] / un, u[1] / un, u[2] / un};
            double const sstar[3] = {R0[0] / cells[0], R0[1] / cells[1], R0[2] / cells[2]};
            double planesN[6][3];
            double planesH[6];
            dsupportPlanes(R0, cells, u, planesN, planesH);

            double helper[3] = {1.0, 0.0, 0.0};
            if(std::abs(n[0]) >= 0.9)
            {
                helper[0] = 0.0;
                helper[1] = 1.0;
            }
            double e1[3] = {
                n[1] * helper[2] - n[2] * helper[1],
                n[2] * helper[0] - n[0] * helper[2],
                n[0] * helper[1] - n[1] * helper[0]};
            double const l1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
            for(int k = 0; k < 3; ++k)
                e1[k] /= l1;
            double const e2[3] = {
                n[1] * e1[2] - n[2] * e1[1],
                n[2] * e1[0] - n[0] * e1[2],
                n[0] * e1[1] - n[1] * e1[0]};

            double acc[3] = {0.0, 0.0, 0.0};
            for(int im = 0; im < 16; ++im)
            {
                double const mu = kMuX[im];
                double const st = std::sqrt(mu * mu < 1.0 ? 1.0 - mu * mu : 0.0);
                double const wmu = kMuW[im];
                for(int ip = 0; ip < 32; ++ip)
                {
                    double const phi = kPi * (kPhX[ip] + 1.0);
                    double const wphi = kPi * kPhW[ip];
                    double const cph = std::cos(phi);
                    double const sph = std::sin(phi);
                    double Om[3];
                    for(int k = 0; k < 3; ++k)
                        Om[k] = mu * n[k] + st * (cph * e1[k] + sph * e2[k]);
                    double rmax = 1e300;
                    for(int f = 0; f < 6; ++f)
                    {
                        double const dn = planesN[f][0] * Om[0] + planesN[f][1] * Om[1]
                            + planesN[f][2] * Om[2];
                        if(dn > 1e-15)
                        {
                            double const cand = planesH[f] / dn;
                            rmax = cand < rmax ? cand : rmax;
                        }
                    }
                    if(!(rmax > 1e-14))
                        continue;
                    double BO[3];
                    dapplyContract(Om, n, gam, BO);
                    BO[0] /= cells[0];
                    BO[1] /= cells[1];
                    BO[2] /= cells[2];

                    double bp[16];
                    int nb = 0;
                    bp[nb++] = 0.0;
                    bp[nb++] = rmax;
                    for(int a = 0; a < 3; ++a)
                    {
                        if(std::abs(BO[a]) < 1e-300)
                            continue;
                        for(int k = 0; k < 4; ++k)
                        {
                            double const r = (sstar[a] - knots[k]) / BO[a];
                            if(r > 1e-14 && r < rmax - 1e-14)
                            {
                                bool dup = false;
                                for(int q = 0; q < nb; ++q)
                                    if(std::abs(bp[q] - r) < 1e-12 * rmax)
                                        dup = true;
                                if(!dup)
                                    bp[nb++] = r;
                            }
                        }
                    }
                    for(int i = 1; i < nb; ++i)
                    {
                        double const key = bp[i];
                        int j = i - 1;
                        while(j >= 0 && bp[j] > key)
                        {
                            bp[j + 1] = bp[j];
                            --j;
                        }
                        bp[j + 1] = key;
                    }
                    double const wa = wmu * wphi;
                    for(int i = 0; i + 1 < nb; ++i)
                    {
                        double const r0 = bp[i];
                        double const r1 = bp[i + 1];
                        if(r1 - r0 <= 1e-15)
                            continue;
                        double const mid = 0.5 * (r0 + r1);
                        double const half = 0.5 * (r1 - r0);
                        for(int ir = 0; ir < 4; ++ir)
                        {
                            double const r = mid + half * kRadX[ir];
                            double const wr = half * kRadW[ir];
                            double s[3];
                            for(int k = 0; k < 3; ++k)
                                s[k] = sstar[k] - r * BO[k];
                            double const q = dtsc(s[0]) * dtsc(s[1]) * dtsc(s[2]);
                            double const w = wa * wr * q;
                            acc[0] += w * Om[0];
                            acc[1] += w * Om[1];
                            acc[2] += w * Om[2];
                        }
                    }
                }
            }
            double tmp[3];
            dapplyContract(acc, n, gam, tmp);
            double const invVol = 1.0 / (cells[0] * cells[1] * cells[2]);
            Erem[0] = invVol * tmp[0];
            Erem[1] = invVol * tmp[1];
            Erem[2] = invVol * tmp[2];
        }

        ALPAKA_FN_ACC inline void dboxTerm(
            double const R0[3],
            double const cells[3],
            double const u[3],
            DeviceParams const& prm,
            double E[3],
            double B[3])
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double Ebox[3] = {0.0, 0.0, 0.0};
            if(uu2 == 0.0)
            {
                double const p[3] = {R0[0], R0[1], R0[2]};
                double const lo[3] = {-1.5 * cells[0], -1.5 * cells[1], -1.5 * cells[2]};
                double const hi[3] = {1.5 * cells[0], 1.5 * cells[1], 1.5 * cells[2]};
                for(int c = 0; c < 3; ++c)
                    Ebox[c] = dboxField8T(p, lo, hi, c);
                B[0] = B[1] = B[2] = 0.0;
            }
            else
            {
                double const gam = std::sqrt(1.0 + uu2);
                double const beta[3] = {u[0] / gam, u[1] / gam, u[2] / gam};
                double const betaH[3] = {
                    u[0] / std::sqrt(uu2),
                    u[1] / std::sqrt(uu2),
                    u[2] / std::sqrt(uu2)};
                int stretchAxis = -1;
                for(int a = 0; a < 3; ++a)
                    if(std::abs(std::abs(betaH[a]) - 1.0) < 1e-12)
                        stretchAxis = a;
                if(stretchAxis >= 0)
                {
                    double const p[3] = {R0[0], R0[1], R0[2]};
                    double lo[3], hi[3];
                    for(int a = 0; a < 3; ++a)
                    {
                        lo[a] = p[a] - 1.5 * cells[a];
                        hi[a] = p[a] + 1.5 * cells[a];
                    }
                    lo[stretchAxis] *= gam;
                    hi[stretchAxis] *= gam;
                    double const tlo[3] = {-hi[0], -hi[1], -hi[2]};
                    double const thi[3] = {-lo[0], -lo[1], -lo[2]};
                    double const zero[3] = {0.0, 0.0, 0.0};
                    for(int c = 0; c < 3; ++c)
                        Ebox[c] = dboxField8T(zero, tlo, thi, c);
                    Ebox[stretchAxis] /= gam;
                    B[0] = beta[1] * Ebox[2] - beta[2] * Ebox[1];
                    B[1] = beta[2] * Ebox[0] - beta[0] * Ebox[2];
                    B[2] = beta[0] * Ebox[1] - beta[1] * Ebox[0];
                }
                else
                {
                    // Oblique: exact field of the Lorentz-contracted support
                    // parallelepiped (polyhedron gravity, see FastNear.hpp).
                    // The oblique order is irrelevant on this path and is kept
                    // only for CLI/settings compatibility.
                    dexactObliqueBox(R0, cells, u, Ebox);
                    B[0] = beta[1] * Ebox[2] - beta[2] * Ebox[1];
                    B[1] = beta[2] * Ebox[0] - beta[0] * Ebox[2];
                    B[2] = beta[0] * Ebox[1] - beta[1] * Ebox[0];
                }
            }
            double const invVolume = 1.0 / (cells[0] * cells[1] * cells[2]);
            E[0] = invVolume * Ebox[0];
            E[1] = invVolume * Ebox[1];
            E[2] = invVolume * Ebox[2];
            B[0] = invVolume * B[0];
            B[1] = invVolume * B[1];
            B[2] = invVolume * B[2];
        }

        ALPAKA_FN_ACC inline void dshapeField(
            double const x_obs[3],
            double const x_src[3],
            double const u[3],
            double const cells[3],
            DeviceParams const& prm,
            double E[3],
            double B[3])
        {
            double const R0[3] = {x_obs[0] - x_src[0], x_obs[1] - x_src[1], x_obs[2] - x_src[2]};
            double const sstar[3] = {R0[0] / cells[0], R0[1] / cells[1], R0[2] / cells[2]};
            double const Wstar = dtsc(sstar[0]) * dtsc(sstar[1]) * dtsc(sstar[2]);
            bool const singular = Wstar > 0.0;

            double const uu2all = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const gamAll = std::sqrt(1.0 + uu2all);

            // Exact s-space evaluator for the selected support cell classes
            // and out-of-support pairs the remainder cannot resolve.
            bool const useExact = droutesExact(R0, cells, u, prm);
            if(useExact)
            {
                dpolyBoxFieldS(R0, cells, u, E);
                double const beta[3] = {u[0] / gamAll, u[1] / gamAll, u[2] / gamAll};
                B[0] = beta[1] * E[2] - beta[2] * E[1];
                B[1] = beta[2] * E[0] - beta[0] * E[2];
                B[2] = beta[0] * E[1] - beta[1] * E[0];
                return;
            }

            if(singular && uu2all > 0.0)
            {
                double const un = std::sqrt(uu2all);
                bool axial = false;
                for(int a = 0; a < 3; ++a)
                    if(std::abs(std::abs(u[a] / un) - 1.0) < 1e-12)
                        axial = true;
                if(!axial)
                {
                    dobliObliqueRayField(R0, cells, u, E);
                    double const gam = std::sqrt(1.0 + uu2all);
                    double const beta[3] = {u[0] / gam, u[1] / gam, u[2] / gam};
                    B[0] = beta[1] * E[2] - beta[2] * E[1];
                    B[1] = beta[2] * E[0] - beta[0] * E[2];
                    B[2] = beta[0] * E[1] - beta[1] * E[0];
                    return;
                }
            }

            double Ebox[3] = {0.0, 0.0, 0.0};
            double Bbox[3] = {0.0, 0.0, 0.0};
            if(singular)
                dboxTerm(R0, cells, u, prm, Ebox, Bbox);

            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const beta[3] = {u[0] / std::sqrt(1.0 + uu2), u[1] / std::sqrt(1.0 + uu2),
                                    u[2] / std::sqrt(1.0 + uu2)};
            double const invg2 = 1.0 / (1.0 + uu2);

            constexpr double kKnots[3][2] = {{-1.5, -0.5}, {-0.5, 0.5}, {0.5, 1.5}};
            double plo[3][4], phi[3][4];
            int npan[3];
            for(int a = 0; a < 3; ++a)
            {
                if(sstar[a] > -1.5 && sstar[a] < 1.5)
                {
                    int m = 0;
                    for(int k = 0; k < 3; ++k)
                    {
                        if(sstar[a] > kKnots[k][0] && sstar[a] < kKnots[k][1])
                        {
                            plo[a][m] = kKnots[k][0];
                            phi[a][m] = sstar[a];
                            ++m;
                            plo[a][m] = sstar[a];
                            phi[a][m] = kKnots[k][1];
                            ++m;
                        }
                        else
                        {
                            plo[a][m] = kKnots[k][0];
                            phi[a][m] = kKnots[k][1];
                            ++m;
                        }
                    }
                    npan[a] = m;
                }
                else
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        plo[a][k] = kKnots[k][0];
                        phi[a][k] = kKnots[k][1];
                    }
                    npan[a] = 3;
                }
            }

            double Erem[3] = {0.0, 0.0, 0.0};
            int const n = prm.order;
            for(int ax = 0; ax < npan[0]; ++ax)
                for(int ay = 0; ay < npan[1]; ++ay)
                    for(int az = 0; az < npan[2]; ++az)
                    {
                        double const mx = 0.5 * (phi[0][ax] - plo[0][ax]);
                        double const my = 0.5 * (phi[1][ay] - plo[1][ay]);
                        double const mz = 0.5 * (phi[2][az] - plo[2][az]);
                        double const cx = 0.5 * (plo[0][ax] + phi[0][ax]);
                        double const cy = 0.5 * (plo[1][ay] + phi[1][ay]);
                        double const cz = 0.5 * (plo[2][az] + phi[2][az]);
                        for(int ix = 0; ix < n; ++ix)
                        {
                            double const sx = cx + mx * prm.xi[ix];
                            double const wx = mx * prm.wi[ix];
                            double const tscx = dtsc(sx);
                            for(int iy = 0; iy < n; ++iy)
                            {
                                double const sy = cy + my * prm.xi[iy];
                                double const wy = wx * my * prm.wi[iy];
                                double const tscxy = tscx * dtsc(sy);
                                for(int iz = 0; iz < n; ++iz)
                                {
                                    double const sz = cz + mz * prm.xi[iz];
                                    double const S = tscxy * dtsc(sz);
                                    double const w = wy * mz * prm.wi[iz] * (S - Wstar);
                                    double const Rx = R0[0] - sx * cells[0];
                                    double const Ry = R0[1] - sy * cells[1];
                                    double const Rz = R0[2] - sz * cells[2];
                                    double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                                    if(uu2 == 0.0)
                                    {
                                        r2 = r2 < 1e-300 ? 1e-300 : r2;
                                        double const f = 1.0 / (r2 * std::sqrt(r2));
                                        Erem[0] += w * f * Rx;
                                        Erem[1] += w * f * Ry;
                                        Erem[2] += w * f * Rz;
                                    }
                                    else
                                    {
                                        double const rb = Rx * beta[0] + Ry * beta[1] + Rz * beta[2];
                                        double D = r2 * invg2 + rb * rb;
                                        D = D < 1e-300 ? 1e-300 : D;
                                        double const f = invg2 / (D * std::sqrt(D));
                                        Erem[0] += w * f * Rx;
                                        Erem[1] += w * f * Ry;
                                        Erem[2] += w * f * Rz;
                                    }
                                }
                            }
                        }
                    }

            E[0] = Wstar * Ebox[0] + Erem[0];
            E[1] = Wstar * Ebox[1] + Erem[1];
            E[2] = Wstar * Ebox[2] + Erem[2];
            B[0] = Wstar * Bbox[0] + (beta[1] * Erem[2] - beta[2] * Erem[1]);
            B[1] = Wstar * Bbox[1] + (beta[2] * Erem[0] - beta[0] * Erem[2]);
            B[2] = Wstar * Bbox[2] + (beta[0] * Erem[1] - beta[1] * Erem[0]);
        }

        //! One component of the normalized point field (Kernel.hpp).
        ALPAKA_FN_ACC inline double dpointComp(
            double Rx,
            double Ry,
            double Rz,
            double const u[3],
            int c)
        {
            double const uu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            double const invG2 = 1.0 / (1.0 + uu2);
            double Ex, Ey, Ez;
            if(uu2 == 0.0)
            {
                double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                double const r3 = r2 * std::sqrt(r2);
                double const f = r3 > 0.0 ? 1.0 / r3 : 0.0;
                Ex = f * Rx;
                Ey = f * Ry;
                Ez = f * Rz;
            }
            else
            {
                double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                double const gamma = std::sqrt(1.0 + uu2);
                double const rb = (Rx * u[0] + Ry * u[1] + Rz * u[2]) / gamma;
                double const D = r2 * invG2 + rb * rb;
                double const f = D > 0.0 ? invG2 / (D * std::sqrt(D)) : 0.0;
                Ex = f * Rx;
                Ey = f * Ry;
                Ez = f * Rz;
            }
            if(c == 0)
                return Ex;
            if(c == 1)
                return Ey;
            if(c == 2)
                return Ez;
            double const gamma = std::sqrt(1.0 + uu2);
            double const bx = u[0] / gamma;
            double const by = u[1] / gamma;
            double const bz = u[2] / gamma;
            if(c == 3)
                return by * Ez - bz * Ey;
            if(c == 4)
                return bz * Ex - bx * Ez;
            return bx * Ey - by * Ex;
        }

        //! 27-point TSC mesh-near stencil contribution (one component) of one
        //! source.  R = (yee - delta) * cells with the integer support offset
        //! delta, so no grid origin is needed.
        ALPAKA_FN_ACC inline double dmeshComp(
            double const xs[3],
            double const urep[3],
            double const cells[3],
            double const nodeBase[3],
            double const sp[3],
            double const yee[3],
            int c)
        {
            double tau[3];
            long lo[3];
            long hi[3];
            for(int a = 0; a < 3; ++a)
            {
                tau[a] = (xs[a] - nodeBase[a]) / sp[a];
                lo[a] = static_cast<long>(::ceil(tau[a] - 1.5));
                hi[a] = static_cast<long>(::floor(tau[a] + 1.5));
            }
            double sum = 0.0;
            for(long ia = lo[0]; ia <= hi[0]; ++ia)
            {
                double const sa = static_cast<double>(ia) - tau[0];
                double const wa = dtsc(sa);
                if(wa == 0.0)
                    continue;
                for(long ib = lo[1]; ib <= hi[1]; ++ib)
                {
                    double const sb = static_cast<double>(ib) - tau[1];
                    double const wb = wa * dtsc(sb);
                    if(wb == 0.0)
                        continue;
                    for(long ic = lo[2]; ic <= hi[2]; ++ic)
                    {
                        double const sc = static_cast<double>(ic) - tau[2];
                        double const wc = wb * dtsc(sc);
                        if(wc == 0.0)
                            continue;
                        double const Rx = (yee[0] - static_cast<double>(ia)) * cells[0];
                        double const Ry = (yee[1] - static_cast<double>(ib)) * cells[1];
                        double const Rz = (yee[2] - static_cast<double>(ic)) * cells[2];
                        sum += wc * dpointComp(Rx, Ry, Rz, urep, c);
                    }
                }
            }
            return sum;
        }

        // ---- kernel --------------------------------------------------------

        struct FastNearKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                double* out6,
                double const* px,
                double const* py,
                double const* pz,
                double const* pux,
                double const* puy,
                double const* puz,
                double const* pq,
                double const* prux,
                double const* pruy,
                double const* pruz,
                bool stencilMesh,
                std::uint32_t nsrc,
                std::uint32_t nx,
                std::uint32_t ny,
                std::uint32_t nz,
                double ox,
                double oy,
                double oz,
                double spx,
                double spy,
                double spz,
                double cx,
                double cy,
                double cz,
                std::uint32_t chunks,
                DeviceParams prm,
                std::uint32_t total) const
            {
                // One thread per (component, node, source chunk): the six
                // staggered components are independent observations, so this
                // keeps per-thread state to one accumulator and multiplies the
                // number of resident blocks six-fold (the per-node work was
                // block-starved on small tiles).
                std::uint32_t const perComp = static_cast<std::uint32_t>(nx * ny * nz) * chunks;
                for(auto [idx] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{total}}))
                {
                    int const c = static_cast<int>(idx / perComp);
                    std::uint32_t const rem = idx - static_cast<std::uint32_t>(c) * perComp;
                    std::uint32_t const node = rem / chunks;
                    std::uint32_t const chunk = rem - node * chunks;
                    std::uint32_t const ii = node % nx;
                    std::uint32_t const jj = (node / nx) % ny;
                    std::uint32_t const kk = node / (nx * ny);

                    double const nodeBase[3] = {
                        ox + static_cast<double>(ii) * spx,
                        oy + static_cast<double>(jj) * spy,
                        oz + static_cast<double>(kk) * spz};
                    double const sp[3] = {spx, spy, spz};
                    double const cells[3] = {cx, cy, cz};
                    double sum = 0.0;
                    for(std::uint32_t q = chunk; q < nsrc; q += chunks)
                    {
                        double const x_src[3] = {px[q], py[q], pz[q]};
                        double const u[3] = {pux[q], puy[q], puz[q]};
                        double const urep[3] = {prux[q], pruy[q], pruz[q]};
                        double const qw = pq[q];
                        double const x_obs[3] = {
                            nodeBase[0] + prm.delta[c][0] * cx,
                            nodeBase[1] + prm.delta[c][1] * cy,
                            nodeBase[2] + prm.delta[c][2] * cz};
                        double E[3], B[3];
                        dshapeField(x_obs, x_src, u, cells, prm, E, B);
                        double comp = c < 3 ? E[c] : B[c - 3];
                        if(stencilMesh)
                            comp -= dmeshComp(x_src, urep, cells, nodeBase, sp, prm.delta[c], c);
                        double const unit = c < 3 ? prm.unitE : prm.unitB;
                        sum += unit * qw * comp;
                    }
                    std::uint32_t const o = static_cast<std::uint32_t>(c) * nx * ny * nz + node;
                    alpaka::onAcc::atomicAdd(acc, &out6[o], sum);
                }
            }
        };

        // ---- device context -------------------------------------------------

        struct Context
        {
            using Selector = decltype(alpaka::onHost::makeDeviceSelector(
                alpaka::api::cuda,
                alpaka::deviceKind::nvidiaGpu));
            using Device = decltype(std::declval<Selector&>().makeDevice(0));
            using Queue = decltype(std::declval<Device&>().makeQueue());

            Device device;
            Queue queue;

            Context() : device(makeDevice()), queue(device.makeQueue()) {}

            static Device makeDevice()
            {
                auto selector = alpaka::onHost::makeDeviceSelector(
                    alpaka::api::cuda,
                    alpaka::deviceKind::nvidiaGpu);
                if(!selector.isAvailable())
                    throw std::runtime_error("fast near-shell: no CUDA device available");
                return selector.makeDevice(0);
            }
        };

        Context& context()
        {
            static Context ctx;
            return ctx;
        }

        template<typename T>
        auto alloc1(Context& ctx, std::uint32_t n)
        {
            return alpaka::onHost::allocUnified<T>(ctx.device, alpaka::Vec<std::uint32_t, 1u>{n});
        }
    } // namespace

    bool fastNearDeviceAvailable()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::api::cuda,
            alpaka::deviceKind::nvidiaGpu);
        return selector.isAvailable();
    }

    std::vector<double> evalFastNearDevice(
        Bunch const& near,
        Bunch const& mesh,
        bool stencilMesh,
        std::size_t nx,
        std::size_t ny,
        std::size_t nz,
        double ox,
        double oy,
        double oz,
        double spx,
        double spy,
        double spz,
        double cx,
        double cy,
        double cz,
        DeviceParams const& params)
    {
        std::size_t const nodes = nx * ny * nz;
        std::vector<double> out(6 * nodes, 0.0);
        std::size_t const nsrc = near.size();
        if(nodes == 0 || nsrc == 0)
            return out;

        Context& ctx = context();
        std::uint32_t const n = static_cast<std::uint32_t>(nsrc);
        auto px = alloc1<double>(ctx, n);
        auto py = alloc1<double>(ctx, n);
        auto pz = alloc1<double>(ctx, n);
        auto pux = alloc1<double>(ctx, n);
        auto puy = alloc1<double>(ctx, n);
        auto puz = alloc1<double>(ctx, n);
        auto pq = alloc1<double>(ctx, n);
        auto prux = alloc1<double>(ctx, n);
        auto pruy = alloc1<double>(ctx, n);
        auto pruz = alloc1<double>(ctx, n);
        auto dout = alloc1<double>(ctx, static_cast<std::uint32_t>(6 * nodes));
        for(std::uint32_t i = 0; i < n; ++i)
        {
            px.data()[i] = near.x[i];
            py.data()[i] = near.y[i];
            pz.data()[i] = near.z[i];
            pux.data()[i] = near.ux[i];
            puy.data()[i] = near.uy[i];
            puz.data()[i] = near.uz[i];
            pq.data()[i] = near.weight[i] * near.charge[i];
            prux.data()[i] = mesh.ux.empty() ? near.ux[i] : mesh.ux[i];
            pruy.data()[i] = mesh.uy.empty() ? near.uy[i] : mesh.uy[i];
            pruz.data()[i] = mesh.uz.empty() ? near.uz[i] : mesh.uz[i];
        }
        for(std::size_t i = 0; i < 6 * nodes; ++i)
            dout.data()[i] = 0.0;

        std::uint32_t const targetThreads = 131072u;
        std::uint32_t chunks = static_cast<std::uint32_t>(
            (targetThreads + nodes - 1u) / nodes);
        if(chunks < 1u)
            chunks = 1u;
        if(chunks > n)
            chunks = n;
        std::uint32_t const total
            = 6u * static_cast<std::uint32_t>(nodes) * chunks;

        std::uint32_t const frame = 256u;
        ctx.queue.enqueue(
            alpaka::onHost::FrameSpec{(total + frame - 1u) / frame, frame},
            FastNearKernel{},
            dout.data(),
            px.data(),
            py.data(),
            pz.data(),
            pux.data(),
            puy.data(),
            puz.data(),
            pq.data(),
            prux.data(),
            pruy.data(),
            pruz.data(),
            stencilMesh,
            n,
            static_cast<std::uint32_t>(nx),
            static_cast<std::uint32_t>(ny),
            static_cast<std::uint32_t>(nz),
            ox,
            oy,
            oz,
            spx,
            spy,
            spz,
            cx,
            cy,
            cz,
            chunks,
            params,
            total);
        alpaka::onHost::wait(ctx.queue);
        for(std::size_t i = 0; i < 6 * nodes; ++i)
            out[i] = dout.data()[i];
        return out;
    }
} // namespace picBunch::pbfast
