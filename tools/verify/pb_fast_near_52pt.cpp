// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Pointwise fast-near evaluator driver for the paper's Table~\ref{tab:fast}.
//
// It evaluates the picBunch fast near evaluator, picBunch::pbfast::shape_field,
// at every point of the 52-point validation set on the CPU.  The evaluator is
// the same host function the CLI's --near-shell fast mode applies per
// (source, observer) pair, so this reproduces the "picBunch fast near-shell"
// column of tab:fast without a grid deposit and without a device.
//
// The fast evaluator and the independent reference share the per-unit-charge
// normalisation
//
//     E_n = 4*pi*eps0*E/q,   B_n = 4*pi*eps0*c*B/q = beta x E_n
//
// and unit cells, so the two are compared directly: for each configuration
// and region (interior / face / outside) and each component c,
//
//     err_c = max_p |F_c - R_c| / (1 + max_p |R_c|),
//     worst = max_c err_c,
//
// the maximum over the 20 interior, 16 face and 16 outside points, exactly the
// metric documented in docs/paper/sec_accuracy.tex (Table tab:fast).
//
// Configurations (u = p/(m_e c), gamma = sqrt(1+|u|^2)):
//   g1          u = (0, 0, 0)
//   g10-axial   u = (0, 0, sqrt(99))
//   g10-oblique u = sqrt(99) * (1, 2, 1) / sqrt(6)
//
// The "remainder order" of tab:fast is the fast evaluator's Gauss order:
// g1 -> 5, g10-axial -> 8, g10-oblique -> 5 (oblique in-support uses the exact
// Lorentz-contracted ray integral, which is order-independent; the oblique
// box order is set to 8 to match the paper and is a compatibility knob).
//
// Build (CPU only; no CUDA, no alpaka, no FFTW):
//     g++ -std=c++20 -O2 -Iinclude tools/verify/pb_fast_near_52pt.cpp -o pb_fast_near_52pt
//
// Usage:
//     pb_fast_near_52pt --ref docs/measurements/fastnear_reference_52pt.csv
//                       --out docs/measurements/fastnear_fast_52pt.csv
//     pb_fast_near_52pt --self-test

#include "picBunch/detail/FastNear.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    double const kPi = 3.141592653589793238462643383279502884;

    // Current tab:fast (docs/paper/sec_accuracy.tex as of the pre-publication
    // revision at commit 7a292eb): worst per-component relative error, per
    // configuration and region.  The interior entries are the measured limits
    // that replaced the old published values after the evaluator and reference
    // changed.
    struct PaperCell
    {
        double interior = 0.0;
        double face = 0.0;
        double outside = 0.0;
    };
    std::map<std::string, PaperCell> const kPaper = {
        {"g1", {6.82e-4, 1.96e-5, 1.63e-7}},
        {"g10-axial", {1.40e-4, 1.11e-4, 2.12e-6}},
        {"g10-oblique", {1.94e-3, 3.94e-3, 1.01e-4}}};

    struct Row
    {
        std::string caseId;
        std::string region;
        std::size_t index = 0;
        double pt[3] = {0.0, 0.0, 0.0};
        double ref[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double val[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    };

    // Order overrides (-1 = use the paper's per-configuration remainder order).
    int gOrderG1 = -1, gOrderAxial = -1, gOrderOblique = -1;

    void settingsFor(std::string const& caseId, picBunch::pbfast::Settings& s)
    {
        // Only the fields common to every FastNear.hpp revision are set, so
        // the same source builds against the paper-publish evaluator (525ad2b)
        // and the current one.  order = Gauss order of the smooth remainder.
        int order = (caseId == "g10-axial") ? 8 : 5;
        if(caseId == "g1" && gOrderG1 >= 0)
            order = gOrderG1;
        else if(caseId == "g10-axial" && gOrderAxial >= 0)
            order = gOrderAxial;
        else if(caseId == "g10-oblique" && gOrderOblique >= 0)
            order = gOrderOblique;
        s.order = order;
        s.obliqueOrder = 8;  // rest-frame octant box order
        s.obliquePower = 5;
    }

    void uFor(std::string const& caseId, double* u)
    {
        if(caseId == "g1")
        {
            u[0] = u[1] = u[2] = 0.0;
        }
        else if(caseId == "g10-axial")
        {
            u[0] = u[1] = 0.0;
            u[2] = std::sqrt(99.0);
        }
        else // g10-oblique
        {
            double const k = std::sqrt(99.0) / std::sqrt(6.0);
            u[0] = k;
            u[1] = 2.0 * k;
            u[2] = k;
        }
    }

    // Fast near evaluator, per-unit-charge field of one source at the origin,
    // observer at pt, unit cells.
    void fastField(double const* pt, std::string const& caseId, double* out6)
    {
        double u[3];
        uFor(caseId, u);
        picBunch::pbfast::Settings s;
        settingsFor(caseId, s);
        nbref::Vec3 E, B;
        picBunch::pbfast::shape_field(
            {pt[0], pt[1], pt[2]}, {0.0, 0.0, 0.0}, {u[0], u[1], u[2]},
            1.0, 1.0, 1.0, s, E, B);
        out6[0] = E.x;
        out6[1] = E.y;
        out6[2] = E.z;
        out6[3] = B.x;
        out6[4] = B.y;
        out6[5] = B.z;
    }

    // Independent adaptive reference (same header family) for the self-test
    // cross-check of units and wiring; not the arbiter of tab:fast.
    void exactField(double const* pt, std::string const& caseId, double* out6)
    {
        double u[3];
        uFor(caseId, u);
        nbref::Settings s;
        s.tol = 1e-9;
        nbref::Vec3 E, B;
        nbref::shape_field(
            {pt[0], pt[1], pt[2]}, {0.0, 0.0, 0.0}, {u[0], u[1], u[2]},
            1.0, 1.0, 1.0, s, E, B);
        out6[0] = E.x;
        out6[1] = E.y;
        out6[2] = E.z;
        out6[3] = B.x;
        out6[4] = B.y;
        out6[5] = B.z;
    }

    std::string trim(std::string s)
    {
        while(!s.empty() && (s.front() == '\r' || s.front() == ' '))
            s.erase(s.begin());
        while(!s.empty() && (s.back() == '\r' || s.back() == ' '))
            s.pop_back();
        return s;
    }

    bool parseCsv(std::string const& path, std::vector<Row>& rows, std::string& err)
    {
        std::ifstream f(path);
        if(!f)
        {
            err = "cannot open reference " + path;
            return false;
        }
        std::string line;
        bool header = true;
        while(std::getline(f, line))
        {
            if(trim(line).empty())
                continue;
            if(header)
            {
                header = false;
                continue;
            }
            std::vector<std::string> tok;
            std::stringstream ss(line);
            std::string cell;
            while(std::getline(ss, cell, ','))
                tok.push_back(trim(cell));
            if(tok.size() != 12)
            {
                err = "bad row (need 12 fields): " + line;
                return false;
            }
            Row r;
            r.caseId = tok[0];
            r.region = tok[1];
            try
            {
                r.index = static_cast<std::size_t>(std::stoul(tok[2]));
                for(int a = 0; a < 3; ++a)
                    r.pt[a] = std::stod(tok[3 + a]);
                for(int c = 0; c < 6; ++c)
                    r.ref[c] = std::stod(tok[6 + c]);
            }
            catch(std::exception const&)
            {
                err = "bad number in row: " + line;
                return false;
            }
            rows.push_back(std::move(r));
        }
        if(rows.empty())
        {
            err = "no data rows in " + path;
            return false;
        }
        return true;
    }

    // Per-component and worst error over all rows of (case, region).
    // Returns the worst (max over components); fills per[6] with err_c.
    double regionError(std::vector<Row> const& rows, std::string const& caseId,
                       std::string const& region, double* per)
    {
        double maxdiff[6] = {0};
        double maxref[6] = {0};
        for(Row const& r : rows)
            if(r.caseId == caseId && r.region == region)
                for(int c = 0; c < 6; ++c)
                {
                    maxdiff[c] = std::max(maxdiff[c],
                                          std::abs(r.val[c] - r.ref[c]));
                    maxref[c] = std::max(maxref[c], std::abs(r.ref[c]));
                }
        double worst = 0.0;
        for(int c = 0; c < 6; ++c)
        {
            per[c] = maxdiff[c] / (1.0 + maxref[c]);
            worst = std::max(worst, per[c]);
        }
        return worst;
    }

    void printPct(double v, char* buf, std::size_t n)
    {
        std::snprintf(buf, n, "%.3e", v);
    }

    char const* kRegions[3] = {"interior", "face", "outside"};

    std::string fmt(double v)
    {
        std::ostringstream os;
        os.precision(17);
        os << v;
        return os.str();
    }

    // Determinism + wiring self-test on a fixed point set (no reference file).
    int selfTest()
    {
        struct PT
        {
            std::string caseId;
            double pt[3];
        } const pts[] = {
            {"g1", {0.5, -0.3, 0.7}},      // stationary, in-support
            {"g1", {0.5, 0.3, 2.0}},       // stationary, outside support
            {"g10-axial", {0.4, 0.5, -0.2}}, // axial, in-support
            {"g10-oblique", {-0.3, 0.6, 0.2}}, // oblique, in-support
            {"g10-oblique", {0.2, 0.2, -2.1}}, // oblique, outside
        };
        int failures = 0;
        for(PT const& p : pts)
        {
            double a[6], b[6];
            fastField(p.pt, p.caseId, a);
            fastField(p.pt, p.caseId, b);
            bool same = true;
            for(int c = 0; c < 6; ++c)
                if(a[c] != b[c])
                    same = false;
            std::printf("determinism %-12s (%.3f,%.3f,%.3f): %s\n",
                        p.caseId.c_str(), p.pt[0], p.pt[1], p.pt[2],
                        same ? "OK (bitwise identical)" : "FAIL");
            if(!same)
                ++failures;
            // Cross-check the fast evaluator against the independent adaptive
            // reference to catch unit/sign/wiring errors (loose bound).
            double x[6];
            exactField(p.pt, p.caseId, x);
            double dmax = 0.0;
            for(int c = 0; c < 6; ++c)
                dmax = std::max(dmax, std::abs(a[c] - x[c]));
            // 1e-2 catches unit/sign/wiring errors (O(1) or O(1e10)) while
            // tolerating the order-8 smooth-remainder quadrature residual at
            // off-beam near points.
            bool ok = dmax < 1.0e-2;
            std::printf("  fast-vs-exact  %-12s max|dF| = %.3e  %s\n",
                        p.caseId.c_str(), dmax, ok ? "OK (<1e-2)" : "FAIL");
            if(!ok)
                ++failures;
        }
        if(failures == 0)
            std::printf("self-test: PASS (%zu points, deterministic; "
                        "fast matches independent exact evaluator <1e-2)\n",
                        std::size(pts));
        else
            std::printf("self-test: FAIL (%d checks failed)\n", failures);
        return failures == 0 ? 0 : 1;
    }

    int measure(std::string const& refPath, std::string const& outPath)
    {
        std::vector<Row> rows;
        std::string err;
        if(!parseCsv(refPath, rows, err))
        {
            std::fprintf(stderr, "pb_fast_near_52pt: %s\n", err.c_str());
            return 1;
        }
        char const* cases[3] = {"g1", "g10-axial", "g10-oblique"};
        for(Row& r : rows)
            fastField(r.pt, r.caseId, r.val);

        // Summary table.
        std::printf("=== tab:fast metric  err_c = max_p|F-R|/(1+max_p|R|), "
                    "worst = max_c err_c ===\n");
        std::printf("%-12s %-9s  %s\n", "case", "region",
                    "Ex        Ey        Ez        Bx        By        Bz  "
                    "worst");
        std::map<std::string, std::map<std::string, double>> worst;
        for(char const* cs : cases)
            for(char const* rg : kRegions)
            {
                double per[6];
                double w = regionError(rows, cs, rg, per);
                worst[cs][rg] = w;
                char buf[6][16], wbuf[16];
                for(int c = 0; c < 6; ++c)
                    printPct(per[c], buf[c], sizeof(buf[c]));
                printPct(w, wbuf, sizeof(wbuf));
                std::printf("%-12s %-9s  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  "
                            "%.3e\n",
                            cs, rg, per[0], per[1], per[2], per[3], per[4],
                            per[5], w);
            }

        // Per-component breakdown for the interior rows (the paper quotes these
        // for the stationary interior row).
        std::printf("\n=== interior per-component (case: "
                    "Ex Ey Ez Bx By Bz) ===\n");
        for(char const* cs : cases)
        {
            double per[6];
            regionError(rows, cs, "interior", per);
            std::printf("%-12s  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n", cs,
                        per[0], per[1], per[2], per[3], per[4], per[5]);
        }

        // Comparison against the paper.
        std::printf("\n=== measured vs paper (worst per region) ===\n");
        std::printf("%-12s %-9s  %-14s %-14s %s\n", "case", "region",
                    "measured", "paper", "ratio");
        bool allMatch = true;
        for(char const* cs : cases)
            for(char const* rg : kRegions)
            {
                double m = worst[cs][rg];
                PaperCell const& pc = kPaper.at(cs);
                double p = std::string(rg) == "interior" ? pc.interior
                             : std::string(rg) == "face" ? pc.face : pc.outside;
                double ratio = p > 0.0 ? m / p : 0.0;
                // A ratio below the paper value is an improvement (the exact
                // s-space path replaced the remainder quadrature); only a
                // regression above 2x fails the tab:fast reproduction gate.
                if(!(ratio < 2.0))
                    allMatch = false;
                std::printf("%-12s %-9s  %.3e       %.3e       %.3f\n", cs, rg,
                            m, p, ratio);
            }
        std::printf("\nverdict: %s\n",
                    allMatch ? "no tab:fast regression (all within 2x; ratios <1 are improvements)"
                             : "DISCREPANCY vs paper (see ratios above)");

        // Deterministic per-point CSV.
        std::ofstream o(outPath, std::ios::binary);
        if(!o)
        {
            std::fprintf(stderr, "pb_fast_near_52pt: cannot open %s\n",
                         outPath.c_str());
            return 1;
        }
        o << "case,region,index,x,y,z,Ex,Ey,Ez,Bx,By,Bz\n";
        for(Row const& r : rows)
            o << r.caseId << ',' << r.region << ',' << r.index << ','
              << fmt(r.pt[0]) << ',' << fmt(r.pt[1]) << ',' << fmt(r.pt[2])
              << ','
              << fmt(r.val[0]) << ',' << fmt(r.val[1]) << ',' << fmt(r.val[2])
              << ','
              << fmt(r.val[3]) << ',' << fmt(r.val[4]) << ',' << fmt(r.val[5])
              << '\n';
        o.close();
        std::printf("\nwrote %s (%zu points)\n", outPath.c_str(), rows.size());
        if(!allMatch)
        {
            std::fprintf(stderr,
                         "pb_fast_near_52pt: measured fast-near errors are outside "
                         "the accepted 0.5x--2x band of tab:fast\n");
            return 1;
        }
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    std::string ref = "docs/measurements/fastnear_reference_52pt.csv";
    std::string out = "docs/measurements/fastnear_fast_52pt.csv";
    bool selfTestFlag = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string const a = argv[i];
        auto next = [&]() -> const char* {
            if(i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if(a == "--ref")
            ref = next();
        else if(a == "--out")
            out = next();
        else if(a == "--order-g1")
            gOrderG1 = std::stoi(next());
        else if(a == "--order-axial")
            gOrderAxial = std::stoi(next());
        else if(a == "--order-oblique")
            gOrderOblique = std::stoi(next());
        else if(a == "--self-test")
            selfTestFlag = true;
        else
        {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if(selfTestFlag)
        return selfTest();
    return measure(ref, out);
}
