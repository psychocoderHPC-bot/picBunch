// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Unit tests for the exact piecewise-polynomial near-field evaluator.
//
// Truth is independent of the evaluator:
//  - the p=1 full-box oracle against exactObliqueBox / the static face
//    integral (closed form, already shipped before this change);
//  - the I_100@dc!=0 moment against a 60-digit direct cubature, with a note on
//    the foot-centred convention that fails it;
//  - the p=2 / p=6 polynomial moments against 60-digit mpmath direct
//    cubature (cone decomposition, 5 observer positions, beta=0.9 and
//    gamma=10).  Reference convergence is <=1e-9 at gamma=10.

#include "picBunch/detail/FastNear.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace
{
    using nbref::Vec3;
    namespace detail = picBunch::pbfast::detail;

    double relErr(Vec3 const& a, Vec3 const& b)
    {
        Vec3 const d = a - b;
        double const n = std::sqrt(nbref::dot(b, b));
        return std::sqrt(nbref::dot(d, d)) / (n > 0.0 ? n : 1.0);
    }

    Vec3 applyMinv(Vec3 const& v, Vec3 const& n, double gam)
    {
        return detail::applyContract(v, n, gam);
    }

    detail::Poly6 makeP2()
    {
        detail::Poly6 p;
        detail::polyZero(p);
        detail::polyAdd(p, 0, 0, 0, 0.7);
        detail::polyAdd(p, 1, 0, 0, -1.3);
        detail::polyAdd(p, 0, 1, 0, 0.4);
        detail::polyAdd(p, 1, 1, 0, 0.9);
        detail::polyAdd(p, 0, 0, 2, 1.7);
        detail::polyAdd(p, 2, 0, 0, 0.2);
        detail::polyAdd(p, 0, 2, 0, -0.5);
        detail::polyAdd(p, 1, 0, 1, 0.3);
        return p;
    }

    detail::Poly6 makeP6()
    {
        detail::Poly6 q0, q1, q2, t, p;
        detail::polyZero(q0);
        detail::polyAdd(q0, 0, 0, 0, 0.5);
        detail::polyAdd(q0, 1, 0, 0, 0.3);
        detail::polyAdd(q0, 0, 1, 0, -0.7);
        detail::polyAdd(q0, 0, 0, 1, 0.2);
        detail::polyAdd(q0, 2, 0, 0, 0.9);
        detail::polyAdd(q0, 1, 1, 0, 0.1);
        detail::polyAdd(q0, 0, 0, 2, 0.4);
        detail::polyZero(q1);
        detail::polyAdd(q1, 0, 0, 0, 1.0);
        detail::polyAdd(q1, 1, 0, 0, -0.4);
        detail::polyAdd(q1, 0, 1, 0, 0.2);
        detail::polyAdd(q1, 0, 0, 1, 0.6);
        detail::polyAdd(q1, 0, 2, 0, 0.5);
        detail::polyAdd(q1, 1, 0, 1, 0.3);
        detail::polyAdd(q1, 0, 0, 2, -0.2);
        detail::polyZero(q2);
        detail::polyAdd(q2, 0, 0, 0, 0.8);
        detail::polyAdd(q2, 1, 0, 0, 0.5);
        detail::polyAdd(q2, 0, 1, 0, -0.3);
        detail::polyAdd(q2, 0, 0, 1, 0.1);
        detail::polyAdd(q2, 0, 0, 2, 0.7);
        detail::polyAdd(q2, 2, 0, 0, 0.25);
        detail::polyAdd(q2, 0, 1, 1, 0.15);
        detail::polyMul(q0, q1, t);
        detail::polyMul(t, q2, p);
        return p;
    }

    //! Static (u = 0) full-support p=1 reference: -sum_f n_f int_face 1/|d| dA.
    Vec3 staticBoxJ(Vec3 const& dc, Vec3 const& cells)
    {
        Vec3 const e[3] = {
            Vec3{3.0 * cells.x, 0.0, 0.0},
            Vec3{0.0, 3.0 * cells.y, 0.0},
            Vec3{0.0, 0.0, 3.0 * cells.z}};
        Vec3 q[6][4];
        Vec3 nrm[6];
        double h[6];
        detail::polyBoxFaces(dc, e, q, nrm, h);
        Vec3 J{};
        for(int f = 0; f < 6; ++f)
            J = J - nrm[f] * detail::boxFaceIntegral(q[f], nrm[f]);
        return J;
    }

    struct PolyCase
    {
        char const* id;
        Vec3 dc;
        Vec3 ref;
    };
    bool startsWithB(char const* id)
    {
        return id[0] == 'b';
    }
    Vec3 const kEb09[3] = {
        {0.52990824529505609, 0.18803670188197755, 0.094018350940988774},
        {0.18803670188197755, 0.81196329811802248, 0.18803670188197755},
        {0.094018350940988774, 0.18803670188197755, 0.52990824529505609}};
    Vec3 const kEg10[3] = {
        {0.25, 0.29999999999999999, 0.14999999999999999},
        {0.29999999999999999, 0.69999999999999996, 0.29999999999999999},
        {0.14999999999999999, 0.29999999999999999, 0.25}};
} // namespace

// ---- gate: p=1 full-support oracle -----------------------------------------
TEST(FastNearPoly, P1MatchesExactObliqueBoxAndStatic)
{
    detail::Poly6 one;
    detail::polyZero(one);
    detail::polyAdd(one, 0, 0, 0, 1.0);
    double const dir[3] = {1.0, 2.0, 1.0};
    double const dl = std::sqrt(6.0);
    double const r0s[][3] = {
        {0.3, 0.2, -0.1},
        {0.5, 0.5, 0.5},
        {2.0, 0.2, -0.1},
        {1.5, 0.2, -0.1}};
    double const cellsSet[][3] = {{1.0, 1.0, 1.0}, {0.7, 1.3, 1.0}};
    for(auto const& c3 : cellsSet)
    {
        Vec3 const cells{c3[0], c3[1], c3[2]};
        for(double gam : {1.0, 10.0, 100.0})
        {
            double const um = std::sqrt(std::max(0.0, gam * gam - 1.0));
            Vec3 const u = um == 0.0
                ? Vec3{}
                : Vec3{um * dir[0] / dl, um * dir[1] / dl, um * dir[2] / dl};
            double const uu2 = nbref::dot(u, u);
            Vec3 const n = uu2 == 0.0 ? Vec3{} : u / std::sqrt(uu2);
            for(auto const& r3 : r0s)
            {
                Vec3 const R0{r3[0] * cells.x, r3[1] * cells.y, r3[2] * cells.z};
                Vec3 J;
                detail::boxPolyJ(
                    R0, cells, u, one, Vec3{0.0, 0.0, 0.0}, Vec3{1.5, 1.5, 1.5}, J);
                Vec3 const E = uu2 == 0.0 ? J : applyMinv(J, n, gam);
                Vec3 const ref = uu2 == 0.0
                    ? staticBoxJ(R0, cells)
                    : detail::exactObliqueBox(R0, cells, u);
                EXPECT_LT(relErr(E, ref), 1.0e-12)
                    << "gam=" << gam << " R0=" << R0.x << "," << R0.y << "," << R0.z;
            }
        }
    }
}

// ---- required review fix: I_100@dc!=0 must use the dc-centred convention ---
TEST(FastNearPoly, VolumeMomentI100UsesDcCentredConvention)
{
    Vec3 const dc{2.0, 1.0, -0.5};
    Vec3 const e[3] = {
        Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
    double I[3][3][3];
    detail::boxVolumeI(dc, e, I);
    // 60-digit direct cubature over [1.5,2.5]x[0.5,1.5]x[-1,0].
    EXPECT_NEAR(I[0][0][0], 0.4364236051369486, 1.0e-12);
    EXPECT_NEAR(I[1][0][0], -0.0137512712141022, 1.0e-12);
    EXPECT_NEAR(I[2][0][0], 0.0366641533876238, 1.0e-12);
    EXPECT_NEAR(I[0][1][0], -0.0070521017595879, 1.0e-12);
    // A foot-centred face convention (the design §4.3/§4.5 mistake) yields
    // I_100 ~ 0.21007 here, so the assertion above fails it by ~2.2e-1.
    EXPECT_GT(std::abs(0.21007378298038984 - (-0.0137512712141022)), 1.0e-1);
}

// ---- gate: p=2 / p=6 against 60-digit mpmath direct cubature ----------------
namespace
{
    PolyCase const kP2Cases[] = {
        {"b09_inside", {0.0, 0.0, 0.0}, {-0.259979509416671695, 0.0633321736085393494, -0.0132820977631860574}},
        {"b09_outside", {1.3247706132376402, 0.47009175470494385, 0.23504587735247193}, {0.0711538170836752276, 0.0213780540373942410, 0.0103348592865180038}},
        {"b09_face", {0.26495412264752805, 0.094018350940988774, 0.047009175470494387}, {1.11559248525270038, -0.116377998969352831, -0.0835435733560782184}},
        {"b09_edge", {0.35897247358851686, 0.5, 0.14102752641148317}, {0.508181287726833908, 0.605039239060012335, -0.116876544793912671}},
        {"b09_corner", {0.40598164905901124, 0.59401835094098876, 0.40598164905901124}, {0.2268389705764730, 0.3074815309419742, 0.2381457888583411}},
        {"g10_inside", {0.0, 0.0, 0.0}, {-0.0178926689307241753, 1.79343986901234922e-05, -0.00347393372179691413}},
        {"g10_outside", {0.625, 0.75, 0.375}, {0.00607748845334919741, 0.00613606417640386731, 0.00303416651445249930}},
        {"g10_face", {0.125, 0.14999999999999999, 0.074999999999999997}, {0.255652692802773251, -0.0726611883840610900, -0.0255039940050839492}},
        {"g10_edge", {0.27500000000000002, 0.5, 0.22500000000000001}, {0.0721299852512700956, 0.0808980681405748608, -0.0575687578276921809}},
        {"g10_corner", {0.34999999999999998, 0.65000000000000002, 0.34999999999999998}, {0.0117187397309731437, 0.0207561814918579765, 0.0117617747321840382}},
    };
    PolyCase const kP6Cases[] = {
        {"b09_inside", {0.0, 0.0, 0.0}, {0.0532933184188167514, -0.164449979986144856, 0.0791423321094298221}},
        {"b09_outside", {1.3247706132376402, 0.47009175470494385, 0.23504587735247193}, {0.0429741544049448004, 0.0112561895575479087, 0.00648507423978567435}},
        {"b09_face", {0.26495412264752805, 0.094018350940988774, 0.047009175470494387}, {0.598138682667922139, -0.162981506846141544, 0.0151478982172280948}},
        {"b09_edge", {0.35897247358851686, 0.5, 0.14102752641148317}, {0.351322614797604738, 0.367391397194987979, -0.0602664573841958234}},
        {"b09_corner", {0.40598164905901124, 0.59401835094098876, 0.40598164905901124}, {0.148614347006129049, 0.180303274736980323, 0.151220715718726501}},
        {"g10_inside", {0.0, 0.0, 0.0}, {0.00114324862537703111, -0.0149481733990437914, 0.00248259876350970243}},
        {"g10_outside", {0.625, 0.75, 0.375}, {0.00385179938252051780, 0.00380115546106652704, 0.00188488887169111310}},
        {"g10_face", {0.125, 0.14999999999999999, 0.074999999999999997}, {0.150308155141501004, -0.0512328399293230097, -0.0118267802529652410}},
        {"g10_edge", {0.27500000000000002, 0.5, 0.22500000000000001}, {0.0499521272489854076, 0.0540403398428900214, -0.0424202094232957599}},
        {"g10_corner", {0.34999999999999998, 0.65000000000000002, 0.34999999999999998}, {0.00788060614984221162, 0.0138317913542105909, 0.00789049899877396210}},
    };

    void checkCases(PolyCase const* cases, detail::Poly6 const& p)
    {
        for(int i = 0; i < 10; ++i)
        {
            PolyCase const& c = cases[i];
            Vec3 const* e = startsWithB(c.id) ? kEb09 : kEg10;
            Vec3 J;
            detail::polyJ(c.dc, e, p, J);
            double const tol = startsWithB(c.id) ? 1.0e-12 : 1.0e-9;
            EXPECT_LT(relErr(J, c.ref), tol) << c.id;
        }
    }
} // namespace

TEST(FastNearPoly, P2MatchesMpmathReference)
{
    checkCases(kP2Cases, makeP2());
}

TEST(FastNearPoly, P6MatchesMpmathReference)
{
    checkCases(kP6Cases, makeP6());
}

// ---- gate: geometric validity (not merely finiteness) ----------------------
TEST(FastNearPoly, GeometricValidityGate)
{
    double const gam = 10.0;
    double const um = std::sqrt(gam * gam - 1.0);
    double const dl = std::sqrt(6.0);
    Vec3 const u{um / dl, 2.0 * um / dl, um / dl};
    Vec3 const cells{1.0, 1.0, 1.0};
    EXPECT_TRUE(detail::polyBoxUsable(Vec3{0.25, 0.4, 0.25}, cells, u));
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{0.25, 0.4, 0.25}, cells, Vec3{}));
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{0.25, 0.5 + 1e-9, 0.25}, cells, u));
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{1.5, 0.2, 0.1}, cells, u));
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{2.0, 0.2, 0.1}, cells, u));
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{1.4999999, 0.2, 0.1}, cells, u));
    // gamma -> 1 (tiny but nonzero u) remains valid when the geometry is fine.
    EXPECT_TRUE(detail::polyBoxUsable(Vec3{0.25, 0.4, 0.25}, cells, Vec3{1e-7, 0.0, 0.0}));
    // uu2 below the 1e-24 static guard is rejected.
    EXPECT_FALSE(detail::polyBoxUsable(Vec3{0.25, 0.4, 0.25}, cells, Vec3{1e-13, 0.0, 0.0}));
}
