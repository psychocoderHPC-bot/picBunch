// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Standalone direct-quadrature reference for the oblique in-support TSC field.
//
// It evaluates the defining integral of the TSC-convolved uniformly moving
// source directly in Lorentz-contracted d-space:
//
//   E(R0) = (1/V) M^{-1} int_{S^2} Omega [ int_0^rmax S(s(r Omega)) dr ] dOmega
//
// with dense Gauss-Legendre rules: `nmu` polar nodes (axis aligned with beta),
// `nphi` azimuth nodes and `nrad` radial nodes on every TSC knot segment.  It
// shares no code with the s-space moment evaluator under test; only the shape
// function and the Lorentz contraction are reimplemented from their
// definitions.  Convergence is demonstrated by running two resolutions.
//
// Usage: ray_reference <nmu> <nphi> <nrad>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    struct Vec
    {
        double x = 0.0, y = 0.0, z = 0.0;
    };

    Vec operator+(Vec a, Vec const& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    Vec operator-(Vec a, Vec const& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    Vec operator*(double s, Vec a) { return {s * a.x, s * a.y, s * a.z}; }
    Vec operator/(Vec a, double s) { return {(1.0 / s) * a.x, (1.0 / s) * a.y, (1.0 / s) * a.z}; }
    double dot(Vec const& a, Vec const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    Vec cross(Vec const& a, Vec const& b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
    double norm(Vec const& v) { return std::sqrt(dot(v, v)); }

    // Definition of the TSC shape (independent of PicBunch headers).
    double tsc(double x)
    {
        double const ax = std::abs(x);
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
            return 0.5 * (1.5 - ax) * (1.5 - ax);
        return 0.0;
    }

    //! v_par + k v_perp about n (n unit).  k = 1/gamma contracts.
    Vec contract(Vec const& v, Vec const& n, double k)
    {
        double const vn = dot(v, n);
        return vn * n + k * (v - vn * n);
    }

    struct Rule
    {
        std::vector<double> x, w;
    };

    Rule gauss(int n)
    {
        Rule r;
        r.x.resize(static_cast<std::size_t>(n));
        r.w.resize(static_cast<std::size_t>(n));
        int const m = (n + 1) / 2;
        for(int i = 0; i < m; ++i)
        {
            double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            double pp = 0.0;
            for(int it = 0; it < 100; ++it)
            {
                double p1 = 1.0, p2 = 0.0;
                for(int j = 1; j <= n; ++j)
                {
                    double const p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                double const dz = p1 / pp;
                z -= dz;
                if(std::abs(dz) < 1e-15)
                    break;
            }
            double const w = 2.0 / ((1.0 - z * z) * pp * pp);
            r.x[static_cast<std::size_t>(i)] = -z;
            r.x[static_cast<std::size_t>(n - 1 - i)] = z;
            r.w[static_cast<std::size_t>(i)] = w;
            r.w[static_cast<std::size_t>(n - 1 - i)] = w;
        }
        return r;
    }

    struct Plane
    {
        Vec n;
        double h = 0.0;
    };

    //! Outward face planes of the contracted support parallelogram; origin is
    //! strictly inside for an in-support observer.
    void supportPlanes(Vec const& R0, Vec const& cells, Vec const& u, Plane planes[6])
    {
        double const uu2 = dot(u, u);
        double const gam = std::sqrt(1.0 + uu2);
        Vec const n = u / std::sqrt(uu2);
        auto const M = [&](Vec const& v) { return contract(v, n, 1.0 / gam); };
        Vec const c0 = M(R0);
        Vec const e[3] = {
            M({1.5 * cells.x, 0.0, 0.0}),
            M({0.0, 1.5 * cells.y, 0.0}),
            M({0.0, 0.0, 1.5 * cells.z})};
        Vec V[2][2][2];
        for(int a = 0; a < 2; ++a)
            for(int b = 0; b < 2; ++b)
                for(int c = 0; c < 2; ++c)
                    V[a][b][c] = c0 + (a ? 1.0 : -1.0) * e[0] + (b ? 1.0 : -1.0) * e[1]
                        + (c ? 1.0 : -1.0) * e[2];
        int const FI[6][4][3] = {
            {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
            {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}},
            {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
            {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
            {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
            {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}};
        for(int f = 0; f < 6; ++f)
        {
            Vec q[4];
            for(int v = 0; v < 4; ++v)
                q[v] = V[FI[f][v][0]][FI[f][v][1]][FI[f][v][2]];
            Vec nn = cross(q[1] - q[0], q[2] - q[0]);
            nn = (1.0 / norm(nn)) * nn;
            Vec const cen = 0.25 * (q[0] + q[1] + q[2] + q[3]);
            if(dot(cen - c0, nn) < 0.0)
                nn = -1.0 * nn;
            planes[f].n = nn;
            planes[f].h = dot(q[0], nn);
        }
    }

    Vec referenceField(
        Vec const& R0,
        Vec const& cells,
        Vec const& u,
        int nmu,
        int nphi,
        int nrad)
    {
        double const uu2 = dot(u, u);
        double const gam = std::sqrt(1.0 + uu2);
        Vec const n = u / std::sqrt(uu2);
        auto const Minv = [&](Vec const& v) { return contract(v, n, gam); };
        Vec const sstar{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
        Plane planes[6];
        supportPlanes(R0, cells, u, planes);

        Vec const helper = std::abs(n.x) < 0.9 ? Vec{1.0, 0.0, 0.0} : Vec{0.0, 1.0, 0.0};
        Vec const e1 = [&] { Vec t = cross(n, helper); return (1.0 / norm(t)) * t; }();
        Vec const e2 = cross(n, e1);

        Rule const muR = gauss(nmu);
        Rule const phR = gauss(nphi);
        Rule const radR = gauss(nrad);
        constexpr double knots[4] = {-1.5, -0.5, 0.5, 1.5};

        Vec acc{};
        for(int im = 0; im < nmu; ++im)
        {
            double const mu = muR.x[static_cast<std::size_t>(im)];
            double const st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            for(int ip = 0; ip < nphi; ++ip)
            {
                double const phi = M_PI * (phR.x[static_cast<std::size_t>(ip)] + 1.0);
                double const wphi = M_PI * phR.w[static_cast<std::size_t>(ip)];
                Vec const Om = mu * n + st * (std::cos(phi) * e1 + std::sin(phi) * e2);
                double rmax = 1e300;
                for(int f = 0; f < 6; ++f)
                {
                    double const dn = dot(planes[f].n, Om);
                    if(dn > 1e-15)
                        rmax = std::min(rmax, planes[f].h / dn);
                }
                if(!(rmax > 1e-14))
                    continue;
                Vec const BO = Minv(Om);
                double const BOa[3] = {BO.x / cells.x, BO.y / cells.y, BO.z / cells.z};
                double const sst[3] = {sstar.x, sstar.y, sstar.z};
                double bp[16];
                int nb = 0;
                bp[nb++] = 0.0;
                bp[nb++] = rmax;
                for(int a = 0; a < 3; ++a)
                {
                    if(std::abs(BOa[a]) < 1e-300)
                        continue;
                    for(int k = 0; k < 4; ++k)
                    {
                        double const r = (sst[a] - knots[k]) / BOa[a];
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
                double const wa = muR.w[static_cast<std::size_t>(im)] * wphi;
                for(int i = 0; i + 1 < nb; ++i)
                {
                    double const r0 = bp[i];
                    double const r1 = bp[i + 1];
                    if(r1 - r0 <= 1e-15)
                        continue;
                    double const mid = 0.5 * (r0 + r1);
                    double const half = 0.5 * (r1 - r0);
                    for(int ir = 0; ir < nrad; ++ir)
                    {
                        double const r = mid + half * radR.x[static_cast<std::size_t>(ir)];
                        double const wr = half * radR.w[static_cast<std::size_t>(ir)];
                        Vec const s{
                            sstar.x - r * BOa[0],
                            sstar.y - r * BOa[1],
                            sstar.z - r * BOa[2]};
                        double const S = tsc(s.x) * tsc(s.y) * tsc(s.z);
                        acc = acc + (wa * wr * S) * Om;
                    }
                }
            }
        }
        Vec const Eraw = Minv(acc);
        double const invV = 1.0 / (cells.x * cells.y * cells.z);
        return invV * Eraw;
    }
} // namespace

int main(int argc, char** argv)
{
    int const nmu = argc > 1 ? std::atoi(argv[1]) : 128;
    int const nphi = argc > 2 ? std::atoi(argv[2]) : 256;
    int const nrad = argc > 3 ? std::atoi(argv[3]) : 10;
    double const gam = 100.0;
    double const beta = std::sqrt(1.0 - 1.0 / (gam * gam));
    Vec const u{gam * beta / std::sqrt(2.0), 0.0, gam * beta / std::sqrt(2.0)};
    Vec const cells{1.0, 1.0, 1.0};
    std::printf("# ray_reference nmu=%d nphi=%d nrad=%d gamma=%g\n", nmu, nphi, nrad, gam);
    std::printf("d,Ex,Ey,Ez\n");
    for(double const d : {0.1, 0.25, 0.5, 1.0})
    {
        Vec const R0{0.6 * d, 0.8 * d, 0.0};
        Vec const E = referenceField(R0, cells, u, nmu, nphi, nrad);
        std::printf("%.17g,%.17g,%.17g,%.17g\n", d, E.x, E.y, E.z);
    }
    return 0;
}
