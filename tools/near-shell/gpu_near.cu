// gpu_near.cu — minimal alpaka (CUDA-first) per-pair TSC-convolved near-field kernel + benchmark.
//
// Run pr24-20261102, task GPU1.
//
// Quantity (S1-algorithm-spec.md sec.1): for one (source, target staggered point, component) pair
// the TSC-convolved uniformly-moving point-charge field
//
//   F_c = integral_{[-1.5,1.5]^3} W(s) Kpoint_c(R0 - s) ds^3 ,   R0 = x_obs - x_src  (cell units,
//                                                                                     cell = 1)
//
// W = Wx Wy Wz is the separable TSC (quadratic B-spline) shape (support [-1.5,1.5]/axis, integral
// 1 per axis). Kpoint is the per-unit-charge field of one uniformly moving charge (the independent
// reference's "existing velocity handling"):
//
//   invg2 = 1/(1+|u|^2)  (= 1-beta^2) ;  D = |R|^2 invg2 + (R.beta)^2
//   E = invg2 R / D^1.5 ;  B = beta x E           (stationary u==0:  E = R/|R|^3,  B = 0)
//
// Two rules:
//   * plain: plain tensor Gauss-Legendre, order 8 per axis (8^3 = 512 nodes), over the whole support.
//     The TSC factor is separable and depends only on the (fixed) nodes, so it is folded into the
//     node weights on the host; the device loop is pure Kpoint + accumulation.
//   * sing:  the independent-reference scheme: split the support at the TSC knots
//            {-1.5,-0.5,0.5,1.5} (W is a polynomial per panel -> integrated exactly) and, when the
//            target projects inside the support on all axes, split the anchor panel at t = R0 and map
//            every panel by s = c + (b-c) tau^5 with Jacobian (b-c) 5 tau^4 that cancels the 1/r^2
//            singularity. Order N per panel (default 8; (3N)^3 = 13824 non-singular, (4N)^3 = 32768 singular).
//
// One thread per (source, node, component) pair; each quadrature slot (node) is evaluated exactly
// once per pair (no redundant slot recomputation). All precision-critical work is double.
//
// Device code never references global constexpr arrays (they are not visible in device code on the
// CUDA backend); the small constant tables (GL nodes/weights, knots, Yee offsets) are passed as
// read-only kernel parameters in device memory.

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace
{
    constexpr int kPlainNodes = 8 * 8 * 8;  // 512 (plain rule, order 8/axis; fixed baseline)

    // Raw Gauss-Legendre order 8 on [-1,1] (numpy.polynomial.legendre.leggauss(8)); sum(wi) = 2.
    constexpr double kXi[8] = {
        -9.60289856497536176e-01, -7.96666477413626728e-01, -5.25532409916328991e-01,
        -1.83434642495649780e-01, 1.83434642495649780e-01, 5.25532409916328991e-01,
        7.96666477413626728e-01, 9.60289856497536176e-01};
    constexpr double kWi[8] = {
        1.01228536290377064e-01, 2.22381034453374427e-01, 3.13706645877886880e-01,
        3.62683783378361657e-01, 3.62683783378361657e-01, 3.13706645877886880e-01,
        2.22381034453374427e-01, 1.01228536290377064e-01};
    constexpr double kKnots[4] = {-1.5, -0.5, 0.5, 1.5};
    // Yee offsets in cell units, order (Ex,Ey,Ez,Bx,By,Bz), row-major 6x3.
    constexpr double kDelta[6][3] = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

    // PIConGPU TSC (quadratic B-spline), argument in cell units.
    ALPAKA_FN_ACC auto tsc(double x) -> double
    {
        double ax = x < 0.0 ? -x : x;
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
        {
            double t = 1.5 - ax;
            return 0.5 * t * t;
        }
        return 0.0;
    }

    // Per-unit-charge field of one uniformly moving charge (independent-reference form).
    ALPAKA_FN_ACC auto pointField6(
        double Rx,
        double Ry,
        double Rz,
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        double E[3],
        double B[3]) -> void
    {
        double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
        if(stationary)
        {
            double const inv = 1.0 / (r2 * std::sqrt(r2));
            E[0] = Rx * inv;
            E[1] = Ry * inv;
            E[2] = Rz * inv;
            B[0] = B[1] = B[2] = 0.0;
            return;
        }
        double const rb = Rx * bx + Ry * by + Rz * bz;
        double const D = r2 * invg2 + rb * rb;
        double const D15 = D * std::sqrt(D);
        double const f = invg2 / D15;
        E[0] = f * Rx;
        E[1] = f * Ry;
        E[2] = f * Rz;
        B[0] = by * E[2] - bz * E[1];
        B[1] = bz * E[0] - bx * E[2];
        B[2] = bx * E[1] - by * E[0];
    }

    void hostDelta(int c, int a, double& d)
    {
        d = kDelta[c][a];
    }

    // Build the 512-node plain tensor (TSC-folded weights). Node = 1.5*xi, weight = 1.5*wi*tsc(node).
    void buildPlainNodes(std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz, std::vector<double>& qw)
    {
        qx.resize(kPlainNodes);
        qy.resize(kPlainNodes);
        qz.resize(kPlainNodes);
        qw.resize(kPlainNodes);
        for(int i = 0; i < 8; ++i)
            for(int j = 0; j < 8; ++j)
                for(int k = 0; k < 8; ++k)
                {
                    int const n = i + 8 * (j + 8 * k);
                    double const sx = 1.5 * kXi[i];
                    double const sy = 1.5 * kXi[j];
                    double const sz = 1.5 * kXi[k];
                    qx[n] = sx;
                    qy[n] = sy;
                    qz[n] = sz;
                    qw[n] = (1.5 * kWi[i] * tsc(sx)) * (1.5 * kWi[j] * tsc(sy)) * (1.5 * kWi[k] * tsc(sz));
                }
    }

    // Host-only Gauss-Legendre nodes/weights of order n on [-1,1], ascending order (Newton on P_n).
    void hostLeggauss(int n, double* x, double* w)
    {
        int const m = (n + 1) / 2;
        for(int i = 0; i < m; ++i)
        {
            double z = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
            double pp = 0.0;
            for(int it = 0; it < 100; ++it)
            {
                double const z1 = z;
                double p0 = 1.0, p1 = z1, p = 0.0;
                for(int j = 2; j <= n; ++j)
                {
                    p = ((2.0 * j - 1.0) * z1 * p1 - (j - 1.0) * p0) / j;
                    p0 = p1;
                    p1 = p;
                }
                pp = static_cast<double>(n) * (z1 * p1 - p0) / (z1 * z1 - 1.0);
                z = z1 - p1 / pp;
                if(std::abs(z - z1) < 1e-15)
                    break;
            }
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }

    // Host CPU evaluation of the plain 512-node rule for one pair (component c), for self-check.
    double cpuPlain(
        double const r0[3],
        int c,
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        double const qx[],
        double const qy[],
        double const qz[],
        double const qw[])
    {
        double ox = r0[0] + kDelta[c][0];
        double oy = r0[1] + kDelta[c][1];
        double oz = r0[2] + kDelta[c][2];
        double sum = 0.0;
        for(int n = 0; n < kPlainNodes; ++n)
        {
            double E[3];
            double B[3];
            pointField6(ox - qx[n], oy - qy[n], oz - qz[n], bx, by, bz, invg2, stationary, E, B);
            sum += qw[n] * ((c < 3) ? E[c] : B[c - 3]);
        }
        return sum;
    }
}  // namespace

// ---------------------------------------------------------------------------
// PLAIN kernel: one thread per (source, node, component) pair, 512-node tensor GL.
// ---------------------------------------------------------------------------
struct PlainKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto out,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& r0,  // (3*nPairs,)  node - source
        alpaka::concepts::IDataSource auto const& comp,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& qx,
        alpaka::concepts::IDataSource auto const& qy,
        alpaka::concepts::IDataSource auto const& qz,
        alpaka::concepts::IDataSource auto const& qw,
        alpaka::concepts::IDataSource auto const& delta,  // (18,) 6x3 Yee offsets
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        std::uint32_t nPairs) const
    {
        for(auto [idx] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{nPairs}}))
        {
            std::uint32_t const i = idx;
            int const c = static_cast<int>(comp[i]);
            double const ox = r0[3 * i + 0] + delta[3 * c + 0];
            double const oy = r0[3 * i + 1] + delta[3 * c + 1];
            double const oz = r0[3 * i + 2] + delta[3 * c + 2];
            double sum = 0.0;
            for(int n = 0; n < kPlainNodes; ++n)
            {
                double E[3];
                double B[3];
                pointField6(ox - qx[n], oy - qy[n], oz - qz[n], bx, by, bz, invg2, stationary, E, B);
                sum += qw[n] * ((c < 3) ? E[c] : B[c - 3]);
            }
            out[i] = sum;
        }
    }
};

// ---------------------------------------------------------------------------
// SING kernel: independent-reference scheme (knot split + power transform), order N per panel.
// ---------------------------------------------------------------------------
struct SingKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto out,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& r0,  // (3*nPairs,)
        alpaka::concepts::IDataSource auto const& comp,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& xi,  // (8,) GL nodes [-1,1]
        alpaka::concepts::IDataSource auto const& wi,  // (8,) GL weights
        alpaka::concepts::IDataSource auto const& knots,  // (4,)
        alpaka::concepts::IDataSource auto const& delta,  // (18,)
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        int order,
        int refine,
        std::uint32_t nPairs) const
    {
        constexpr double kP = 5.0;
        int const nseg = 1 << refine;  // panel bisections (matches indfield refine)
        int const po = order * nseg;  // nodes per panel
        for(auto [idx] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{nPairs}}))
        {
            std::uint32_t const i = idx;
            int const c = static_cast<int>(comp[i]);
            // anchor = staggered R0 = (x_obs - x_src), matching indfield (s_star = R0/cell).
            double const ox = r0[3 * i + 0] + delta[3 * c + 0];
            double const oy = r0[3 * i + 1] + delta[3 * c + 1];
            double const oz = r0[3 * i + 2] + delta[3 * c + 2];
            double const rr[3] = {ox, oy, oz};
            bool const singular = (std::abs(ox) < 1.5) && (std::abs(oy) < 1.5) && (std::abs(oz) < 1.5);
            double pc[3][4];
            double pb[3][4];
            double pab[3][4];
            int pn[3];
            for(int a = 0; a < 3; ++a)
            {
                int m = 0;
                for(int p = 0; p < 3; ++p)
                {
                    double const lo = knots[p];
                    double const hi = knots[p + 1];
                    auto add = [&](double a0, double b0)
                    {
                        double cn, on;
                        if(singular)
                        {
                            if(std::abs(a0 - rr[a]) <= std::abs(b0 - rr[a]))
                            {
                                cn = a0;
                                on = b0;
                            }
                            else
                            {
                                cn = b0;
                                on = a0;
                            }
                        }
                        else
                        {
                            cn = a0;
                            on = b0;
                        }
                        pc[a][m] = cn;
                        pb[a][m] = on;
                        pab[a][m] = std::abs(on - cn);
                        ++m;
                    };
                    if(singular && lo < rr[a] && rr[a] < hi)
                    {
                        add(lo, rr[a]);
                        add(rr[a], hi);
                    }
                    else
                    {
                        add(lo, hi);
                    }
                }
                pn[a] = m;
            }
            double sum = 0.0;
            int const NX = pn[0] * po, NY = pn[1] * po, NZ = pn[2] * po;
            if(!singular)
            {
                for(int ii = 0; ii < NX; ++ii)
                    for(int jj = 0; jj < NY; ++jj)
                        for(int kk = 0; kk < NZ; ++kk)
                        {
                            double sx, sy, sz, wx, wy, wz;
                            {
                                int const g = ii % order, seg = (ii / order) % nseg, p = ii / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sx = pc[0][p] + (pb[0][p] - pc[0][p]) * t;
                                wx = wi[g] * (0.5 / nseg) * pab[0][p] * tsc(sx);
                            }
                            {
                                int const g = jj % order, seg = (jj / order) % nseg, p = jj / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sy = pc[1][p] + (pb[1][p] - pc[1][p]) * t;
                                wy = wi[g] * (0.5 / nseg) * pab[1][p] * tsc(sy);
                            }
                            {
                                int const g = kk % order, seg = (kk / order) % nseg, p = kk / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sz = pc[2][p] + (pb[2][p] - pc[2][p]) * t;
                                wz = wi[g] * (0.5 / nseg) * pab[2][p] * tsc(sz);
                            }
                            double E[3];
                            double B[3];
                            pointField6(ox - sx, oy - sy, oz - sz, bx, by, bz, invg2, stationary, E, B);
                            sum += (wx * wy * wz) * ((c < 3) ? E[c] : B[c - 3]);
                        }
            }
            else
            {
                for(int ii = 0; ii < NX; ++ii)
                    for(int jj = 0; jj < NY; ++jj)
                        for(int kk = 0; kk < NZ; ++kk)
                        {
                            double sx, sy, sz, wx, wy, wz;
                            {
                                int const g = ii % order, seg = (ii / order) % nseg, p = ii / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sx = pc[0][p] + (pb[0][p] - pc[0][p]) * t5;
                                wx = wi[g] * (0.5 / nseg) * pab[0][p] * kP * t4 * tsc(sx);
                            }
                            {
                                int const g = jj % order, seg = (jj / order) % nseg, p = jj / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sy = pc[1][p] + (pb[1][p] - pc[1][p]) * t5;
                                wy = wi[g] * (0.5 / nseg) * pab[1][p] * kP * t4 * tsc(sy);
                            }
                            {
                                int const g = kk % order, seg = (kk / order) % nseg, p = kk / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sz = pc[2][p] + (pb[2][p] - pc[2][p]) * t5;
                                wz = wi[g] * (0.5 / nseg) * pab[2][p] * kP * t4 * tsc(sz);
                            }
                            double E[3];
                            double B[3];
                            pointField6(ox - sx, oy - sy, oz - sz, bx, by, bz, invg2, stationary, E, B);
                            sum += (wx * wy * wz) * ((c < 3) ? E[c] : B[c - 3]);
                        }
            }
            out[i] = sum;
        }
    }
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
namespace
{
    struct Args
    {
        const char* kernel = "plain";  // plain | sing
        const char* band = "mix";  // in | band | mix
        int order = 8;  // GL order per axis (sing rule); plain rule is fixed at 8
        int refine = 0;  // panel bisections (2^refine subpanels); sing rule only
        std::uint32_t npairs = 1u << 20;
        int iters = 4;
        double gamma = 10.0;
        std::uint64_t seed = 0x51ed1234ULL;
        bool sample = false;
        bool bench = true;
        bool check = true;
        const char* outpath = "gpu1_sample";
    };

    void parseArgs(int argc, char** argv, Args& a)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            auto next = [&](const char* def) -> const char*
            {
                return (i + 1 < argc) ? argv[++i] : def;
            };
            if(s == "--kernel")
                a.kernel = next("plain");
            else if(s == "--band")
                a.band = next("mix");
            else if(s == "--order")
                a.order = std::stoi(next("8"));
            else if(s == "--refine")
                a.refine = std::stoi(next("0"));
            else if(s == "--npairs")
                a.npairs = static_cast<std::uint32_t>(std::stoul(next("0")));
            else if(s == "--iters")
                a.iters = std::stoi(next("4"));
            else if(s == "--gamma")
                a.gamma = std::stod(next("10"));
            else if(s == "--seed")
                a.seed = std::stoull(next("0x51ed1234"));
            else if(s == "--sample")
                a.sample = true;
            else if(s == "--nobs")
                a.bench = false;
            else if(s == "--nocheck")
                a.check = false;
            else if(s == "--out")
                a.outpath = next("gpu1_sample");
        }
    }

    void velocity(double gamma, double u[3], double beta[3], double& invg2, bool& stationary)
    {
        if(gamma <= 1.0)
        {
            u[0] = u[1] = u[2] = 0.0;
            beta[0] = beta[1] = beta[2] = 0.0;
            invg2 = 1.0;
            stationary = true;
            return;
        }
        u[0] = u[1] = 0.0;
        u[2] = std::sqrt(gamma * gamma - 1.0);
        beta[0] = beta[1] = 0.0;
        beta[2] = u[2] / gamma;
        invg2 = 1.0 / (1.0 + u[2] * u[2]);
        stationary = false;
    }

    void genPairs(
        std::uint32_t n,
        const char* band,
        std::uint64_t seed,
        std::vector<double>& rel,
        std::vector<int>& comp)
    {
        rel.resize(3 * static_cast<std::size_t>(n));
        comp.resize(n);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for(std::uint32_t i = 0; i < n; ++i)
        {
            int c = static_cast<int>(i % 6);
            bool inBand;
            if(std::string(band) == "in")
                inBand = true;
            else if(std::string(band) == "band")
                inBand = false;
            else
                inBand = (i % 2 == 0);
            double rx, ry, rz;
            if(inBand)
            {
                rx = 1.5 * uni(rng);
                ry = 1.5 * uni(rng);
                rz = 1.5 * uni(rng);
            }
            else
            {
                // 1.5-4 cell band, but OUTSIDE the TSC support box (|rel|_inf >= 1.5) so the
                // quadrature is non-singular (3 panels/axis, 13824 nodes). Reject diagonal
                // in-box samples.
                for(int attempt = 0; attempt < 64; ++attempt)
                {
                    double gx = gauss(rng), gy = gauss(rng), gz = gauss(rng);
                    double len = std::sqrt(gx * gx + gy * gy + gz * gz);
                    double rad = 1.5 + 2.5 * uni01(rng);
                    rx = rad * gx / len;
                    ry = rad * gy / len;
                    rz = rad * gz / len;
                    if(std::max({std::abs(rx), std::abs(ry), std::abs(rz)}) >= 1.5)
                        break;
                }
            }
            rel[3 * i] = rx;
            rel[3 * i + 1] = ry;
            rel[3 * i + 2] = rz;
            comp[i] = c;
        }
    }

    // flat 6x3 Yee offsets
    double flatDelta[18];
}  // namespace

int main(int argc, char** argv)
{
    using namespace alpaka;
    Args a;
    parseArgs(argc, argv, a);

    for(int c = 0; c < 6; ++c)
        for(int t = 0; t < 3; ++t)
            flatDelta[3 * c + t] = kDelta[c][t];

    double u[3];
    double beta[3];
    double invg2;
    bool stationary;
    velocity(a.gamma, u, beta, invg2, stationary);

    bool const isPlain = std::string(a.kernel) == "plain";
    int const nseg = 1 << a.refine;
    int const perAxis = isPlain ? 8 : (std::string(a.band) == "band" ? 3 * nseg * a.order : 4 * nseg * a.order);
    int const nodesPerPair = perAxis * perAxis * perAxis;  // (panels*nseg*order)^3

    std::printf("gpu_near kernel=%s band=%s order=%d refine=%d gamma=%.4g npairs=%u iters=%d seed=0x%llx\n",
                a.kernel, a.band, a.order, a.refine, a.gamma, a.npairs, a.iters, static_cast<unsigned long long>(a.seed));
    std::printf("  u=(%.4g,%.4g,%.4g) beta=(%.4g,%.4g,%.4g) invg2=%.6g stationary=%d nodes/pair~%d\n",
                u[0], u[1], u[2], beta[0], beta[1], beta[2], invg2, (int)stationary, nodesPerPair);

    std::vector<double> rel;
    std::vector<int> comp;
    genPairs(a.npairs, a.band, a.seed, rel, comp);
    std::vector<double> hOut(a.npairs, 0.0);
    std::vector<double> qx, qy, qz, qw;
    if(isPlain)
        buildPlainNodes(qx, qy, qz, qw);

    auto const deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto devSel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSel.makeDevice(0);
    std::printf("  device: %s\n", onHost::getName(device).c_str());
    onHost::Queue queue = device.makeQueue();

    auto dRel = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{3 * a.npairs});
    auto dComp = onHost::alloc<int>(device, Vec<std::uint32_t, 1u>{a.npairs});
    auto dOut = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{a.npairs});
    auto hRel = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{3 * a.npairs});
    auto hComp = onHost::allocHost<int>(Vec<std::uint32_t, 1u>{a.npairs});
    auto hOutV = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{a.npairs});
    for(std::uint32_t i = 0; i < 3 * a.npairs; ++i)
        hRel[i] = rel[i];
    for(std::uint32_t i = 0; i < a.npairs; ++i)
        hComp[i] = comp[i];
    onHost::memcpy(queue, dRel, hRel);
    onHost::memcpy(queue, dComp, hComp);

    // small constant tables to device
    auto dDelta = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{18});
    auto hDelta = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{18});
    for(int i = 0; i < 18; ++i)
        hDelta[i] = flatDelta[i];
    onHost::memcpy(queue, dDelta, hDelta);
    std::vector<double> hXiV(a.order), hWiV(a.order);
    hostLeggauss(a.order, hXiV.data(), hWiV.data());
    auto dXi = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{a.order});
    auto hXi = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{a.order});
    for(int i = 0; i < a.order; ++i)
        hXi[i] = hXiV[i];
    onHost::memcpy(queue, dXi, hXi);
    auto dWi = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{a.order});
    auto hWi = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{a.order});
    for(int i = 0; i < a.order; ++i)
        hWi[i] = hWiV[i];
    onHost::memcpy(queue, dWi, hWi);
    auto dKnots = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{4});
    auto hKnots = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{4});
    for(int i = 0; i < 4; ++i)
        hKnots[i] = kKnots[i];
    onHost::memcpy(queue, dKnots, hKnots);

    auto const exec = exec::gpuCuda;
    auto const frame = onHost::FrameSpec{divCeil(a.npairs, 256u), 256u, exec};

    auto dQx = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{kPlainNodes});
    auto dQy = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{kPlainNodes});
    auto dQz = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{kPlainNodes});
    auto dQw = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{kPlainNodes});
    if(isPlain)
    {
        auto hQx = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{kPlainNodes});
        auto hQy = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{kPlainNodes});
        auto hQz = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{kPlainNodes});
        auto hQw = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{kPlainNodes});
        for(int i = 0; i < kPlainNodes; ++i)
        {
            hQx[i] = qx[i];
            hQy[i] = qy[i];
            hQz[i] = qz[i];
            hQw[i] = qw[i];
        }
        onHost::memcpy(queue, dQx, hQx);
        onHost::memcpy(queue, dQy, hQy);
        onHost::memcpy(queue, dQz, hQz);
        onHost::memcpy(queue, dQw, hQw);
    }

    auto launch = [&]()
    {
        if(isPlain)
            queue.enqueue(frame, PlainKernel{}, dOut, dRel, dComp, dQx, dQy, dQz, dQw, dDelta,
                          beta[0], beta[1], beta[2], invg2, stationary, a.npairs);
        else
            queue.enqueue(frame, SingKernel{}, dOut, dRel, dComp, dXi, dWi, dKnots, dDelta,
                          beta[0], beta[1], beta[2], invg2, stationary, a.order, a.refine, a.npairs);
    };

    // warm-up
    launch();
    onHost::wait(queue);
    onHost::memcpy(queue, hOutV, dOut);
    onHost::wait(queue);
    for(std::uint32_t i = 0; i < a.npairs; ++i)
        hOut[i] = hOutV[i];

    if(a.check && isPlain)
    {
        std::uint32_t const nc = std::min<std::uint32_t>(64, a.npairs);
        double maxerr = 0.0;
        for(std::uint32_t i = 0; i < nc; ++i)
        {
            double r0[3] = {rel[3 * i], rel[3 * i + 1], rel[3 * i + 2]};
            double ref = cpuPlain(r0, comp[i], beta[0], beta[1], beta[2], invg2, stationary,
                                  qx.data(), qy.data(), qz.data(), qw.data());
            double scale = 1.0 + std::max(std::abs(ref), std::abs(hOut[i]));
            maxerr = std::max(maxerr, std::abs(ref - hOut[i]) / scale);
        }
        std::printf("  self-check (kernel vs CPU plain, %u pairs): max rel err = %.3e\n", nc, maxerr);
    }

    double pairsPerSec = 0.0;
    if(a.bench)
    {
        auto const t0 = std::chrono::steady_clock::now();
        for(int it = 0; it < a.iters; ++it)
            launch();
        onHost::wait(queue);
        auto const t1 = std::chrono::steady_clock::now();
        double const secs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;
        pairsPerSec = static_cast<double>(a.npairs) * a.iters / secs;
        double const nodeRate = pairsPerSec * nodesPerPair;
        std::printf("  bench: %u pairs x %d iters in %.4f s -> %.4e pairs/s (%.4e nodes/s)  [%.3f us/pair]\n",
                    a.npairs, a.iters, secs, pairsPerSec, nodeRate, 1e6 / (pairsPerSec ? pairsPerSec : 1.0));
    }

    if(a.sample)
    {
        std::string base = a.outpath;
        {
            std::ofstream b(base + ".bin", std::ios::binary);
            b.write(reinterpret_cast<char const*>(rel.data()), 3 * static_cast<std::size_t>(a.npairs) * sizeof(double));
            b.write(reinterpret_cast<char const*>(comp.data()), static_cast<std::size_t>(a.npairs) * sizeof(int));
            b.write(reinterpret_cast<char const*>(hOut.data()), static_cast<std::size_t>(a.npairs) * sizeof(double));
        }
        {
            std::ofstream m(base + ".meta.txt");
            m << "kernel " << a.kernel << "\n";
            m << "band " << a.band << "\n";
            m << "npairs " << a.npairs << "\n";
            m << "gamma " << a.gamma << "\n";
            m << "ux " << u[0] << "\nuy " << u[1] << "\nuz " << u[2] << "\n";
            m << "cell 1 1 1\n";
        }
        std::printf("  sample -> %s.bin (+.meta.txt), %u pairs\n", base.c_str(), a.npairs);
    }

    std::printf("DONE kernel=%s band=%s pairsPerSec=%.4e\n", a.kernel, a.band, pairsPerSec);
    return 0;
}
