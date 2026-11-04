// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera
//
// Driver for the standalone ReferenceKernel port.  Host-only; no GPU, no
// openPMD, no production code path.
//
// Modes
//   probes       five hardcoded inSupportAccuracy() probes, upstream defaults
//   convergence  box_field and remainder order convergence at interior/face
//   accuracy     fixed-order-8 field at inside/face/outside points, read from
//                a whitespace table with an independent reference appended
//   bench        per-call cost of the port versus the vendored reference
//
// accuracy table columns:
//   label x y z ux uy uz  Ex Ey Ez Bx By Bz
// where the trailing six are the converged reference (indfield.py).

#include "picBunch/detail/NearReference.hpp"
#include "picBunch/detail/ReferenceKernelPort.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace rk = picBunch::refkernel;

struct Probe {
  char const *name;
  rk::Vec3 x;
  rk::Vec3 truth;
};

void printVec(rk::Vec3 const &v) {
  std::printf("(%.17g, %.17g, %.17g)", v.x, v.y, v.z);
}

double maxAbs(rk::Vec3 const &v) {
  return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
}

int probes() {
  std::array<Probe, 5> const probes{{
      {"P1", {0.5, 0.25, 0.0}, {0.68199314198371386, 0.33013571230043481, 0.0}},
      {"P2",
       {0.1, 0.2, 0.3},
       {0.15460951156243138, 0.31166497467576837, 0.47331363061538945}},
      {"H1",
       {0.49, 0.49, 0.49},
       {0.45480399008387334, 0.45480399008387334, 0.45480399008387334}},
      {"H3",
       {1.49, 0.25, 0.0},
       {0.42013343859130897, 0.066759292829999553, 0.0}},
      {"H6",
       {0.99, 0.99, 0.99},
       {0.19828951566260772, 0.19828951566260772, 0.19828951566260772}},
  }};

  double worst = 0.0;
  int failures = 0;
  std::printf(
      "probe            E_x                      E_y                      E_z  "
      "                    rel        conv  box   rem  nodes\n");
  for (auto const &p : probes) {
    auto const r = rk::in_support_stationary_adaptive(p.x, rk::Vec3{},
                                                      rk::Vec3{1.0, 1.0, 1.0});
    double const scale = std::max({std::abs(p.truth.x), std::abs(p.truth.y),
                                   std::abs(p.truth.z), 1e-300});
    double const diff = maxAbs(r.E - p.truth);
    double const rel = diff / scale;
    worst = std::max(worst, rel);
    if (rel > 1e-6)
      ++failures;
    std::printf("%-6s ", p.name);
    printVec(r.E);
    std::printf("  %.3e  %d  %4d  %4d  %zu\n", rel, r.converged ? 1 : 0,
                r.box_order, r.remainder_order, r.total_node_count);
  }
  std::printf("worst relative error = %.3e; failures = %d\n", worst, failures);
  return failures == 0 ? 0 : 1;
}

double rel6(std::array<double, 6> const &a, std::array<double, 6> const &b) {
  double maxDiff = 0.0;
  double maxVal = 0.0;
  for (int i = 0; i < 6; ++i) {
    maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
    maxVal = std::max(maxVal, std::abs(a[i]));
    maxVal = std::max(maxVal, std::abs(b[i]));
  }
  return maxDiff / (1.0 + maxVal);
}

std::array<double, 6> box6(rk::Vec3 const &R0, int n) {
  rk::Vec3 const v = rk::box_field(R0, rk::Vec3{}, rk::Vec3{1.0, 1.0, 1.0}, n);
  return {v.x, v.y, v.z, 0.0, 0.0, 0.0};
}

std::array<double, 6> rem6(rk::Vec3 const &R0, int n) {
  double const t[3] = {R0.x, R0.y, R0.z};
  double const Wt =
      rk::tsc_shape(t[0]) * rk::tsc_shape(t[1]) * rk::tsc_shape(t[2]);
  rk::Vec3 E, B;
  rk::remainder_support_integral(R0, rk::Vec3{}, rk::Vec3{1.0, 1.0, 1.0}, n, Wt,
                                 E, B);
  return {E.x, E.y, E.z, B.x, B.y, B.z};
}

int convergence() {
  rk::Vec3 const cells{1.0, 1.0, 1.0};
  rk::Vec3 const interior{0.3, 0.2, 0.1};
  rk::Vec3 const face{1.5, 0.4, 0.3};

  std::printf("box_field convergence (relative to n=1024)\n");
  std::printf("  n      interior_err   face_err      face_ratio\n");
  double prevFace = 0.0;
  for (int n : {4, 8, 16, 32, 64, 128}) {
    auto const refI = box6(interior, 1024);
    auto const refF = box6(face, 1024);
    double const eI = rel6(box6(interior, n), refI);
    double const eF = rel6(box6(face, n), refF);
    std::printf("  %-6d %.6e   %.6e", n, eI, eF);
    if (prevFace > 0.0 && eF > 0.0)
      std::printf("   %.3f", prevFace / eF);
    std::printf("\n");
    prevFace = eF;
  }

  std::printf("\nremainder convergence (relative to n=512)\n");
  std::printf("  n      interior_err   face_err      face_ratio\n");
  prevFace = 0.0;
  for (int n : {4, 8, 16, 32, 64}) {
    auto const refI = rem6(interior, 512);
    auto const refF = rem6(face, 512);
    double const eI = rel6(rem6(interior, n), refI);
    double const eF = rel6(rem6(face, n), refF);
    std::printf("  %-6d %.6e   %.6e", n, eI, eF);
    if (prevFace > 0.0 && eF > 0.0)
      std::printf("   %.3f", prevFace / eF);
    std::printf("\n");
    prevFace = eF;
  }

  std::printf("\nnodes per stationary pair\n");
  std::printf("  order   box(6n^2)   remainder(octants*n^3)   total (interior, "
              "8 octants)\n");
  for (int n : {8, 16})
    std::printf("  %-6d %-11zu %-24zu %zu\n", n, rk::box_nodes(n),
                rk::remainder_nodes(interior, cells, n),
                rk::box_nodes(n) + rk::remainder_nodes(interior, cells, n));
  std::printf("  upstream default box order 256 = %zu box nodes\n",
              rk::box_nodes(256));
  return 0;
}

struct Row {
  std::string label;
  rk::Vec3 x{};
  rk::Vec3 u{};
  rk::Vec3 refE{};
  rk::Vec3 refB{};
};

std::vector<Row> readRows(std::string const &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  std::vector<Row> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream s(line);
    Row r;
    double vals[12];
    if (!(s >> r.label))
      continue;
    bool ok = true;
    for (double &v : vals)
      ok = ok && static_cast<bool>(s >> v);
    if (!ok)
      continue;
    r.x = {vals[0], vals[1], vals[2]};
    r.u = {vals[3], vals[4], vals[5]};
    r.refE = {vals[6], vals[7], vals[8]};
    r.refB = {vals[9], vals[10], vals[11]};
    rows.push_back(r);
  }
  return rows;
}

int accuracy(std::string const &path) {
  auto const rows = readRows(path);
  if (rows.empty())
    throw std::runtime_error("no data rows in " + path);
  int const order = 8;
  std::printf("fixed order %d (box order %d, remainder order %d)\n", order,
              order, order);
  std::printf(
      "row                 port_err   cur_err    port_conv  nodes   port_E\n");
  double worstPort = 0.0;
  for (auto const &row : rows) {
    auto const port = rk::in_support_stationary(
        row.x, row.u, rk::Vec3{1.0, 1.0, 1.0}, order, order);
    nbref::Vec3 E, B;
    nbref::ShapeResult sr;
    nbref::shape_field(nbref::Vec3{row.x.x, row.x.y, row.x.z}, nbref::Vec3{},
                       nbref::Vec3{row.u.x, row.u.y, row.u.z}, 1.0, 1.0, 1.0,
                       nbref::Settings{}, E, B, &sr);
    std::array<double, 6> const p{port.E.x, port.E.y, port.E.z,
                                  port.B.x, port.B.y, port.B.z};
    std::array<double, 6> const c{E.x, E.y, E.z, B.x, B.y, B.z};
    std::array<double, 6> const r{row.refE.x, row.refE.y, row.refE.z,
                                  row.refB.x, row.refB.y, row.refB.z};
    double const ep = rel6(p, r);
    double const ec = rel6(c, r);
    worstPort = std::max(worstPort, ep);
    std::printf("%-18s  %.3e  %.3e  %d          %-7zu ", row.label.c_str(), ep,
                ec, sr.converged ? 1 : 0, port.total_node_count);
    printVec(port.E);
    std::printf("\n");
  }
  std::printf("worst port relative error = %.3e\n", worstPort);
  return 0;
}

template <typename Fn> double timeCalls(Fn &&fn, int reps) {
  auto const start = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i)
    fn();
  auto const stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

int bench() {
  rk::Vec3 const cells{1.0, 1.0, 1.0};
  std::vector<rk::Vec3> points;
  for (double x : {0.0, 0.4, 0.9, 1.49})
    for (double y : {0.0, 0.6})
      for (double z : {0.0, 0.3})
        points.push_back({x, y, z});
  auto runPorted = [&](int boxOrder, int remOrder, int reps) {
    double const sec = timeCalls(
        [&]() {
          for (auto const &p : points) {
            auto const r = rk::in_support_stationary(p, rk::Vec3{}, cells,
                                                     boxOrder, remOrder);
            if (!std::isfinite(r.E.x))
              throw std::runtime_error("non-finite");
          }
        },
        reps);
    return sec /
           (static_cast<double>(reps) * static_cast<double>(points.size()));
  };

  std::vector<rk::Vec3> outside;
  for (double x : {2.0, 2.6, 3.2, 4.0})
    for (double y : {0.0, 1.1})
      for (double z : {0.0, 0.7})
        outside.push_back({x, y, z});
  auto runCurrentPoints = [&](std::vector<rk::Vec3> const &ps, int reps) {
    nbref::Settings s;
    s.order = 4;
    s.tol = 1e-7;
    s.max_level = 5;
    s.max_order = 16;
    s.sphere_n = 10;
    s.sphere_max_n = 32;
    s.sphere_radial_order = 4;
    double const sec = timeCalls(
        [&]() {
          for (auto const &p : ps) {
            nbref::Vec3 E, B;
            nbref::shape_field(nbref::Vec3{p.x, p.y, p.z}, nbref::Vec3{},
                               nbref::Vec3{}, 1.0, 1.0, 1.0, s, E, B);
            if (!std::isfinite(E.x))
              throw std::runtime_error("non-finite");
          }
        },
        reps);
    return sec / (static_cast<double>(reps) * static_cast<double>(ps.size()));
  };
  std::printf("per shape_field call, single thread\n");
  std::printf(
      "  current vendored, %zu in-support points : %.4e s/pair  (40 reps)\n",
      points.size(), runCurrentPoints(points, 40));
  std::printf(
      "  current vendored, %zu out-of-support pts: %.4e s/pair  (200 reps)\n",
      outside.size(), runCurrentPoints(outside, 200));
  std::printf("  port box=8  rem=8 : %.4e s/pair  nodes/call=%zu  (20 reps)\n",
              runPorted(8, 8, 20),
              rk::box_nodes(8) + rk::remainder_nodes(points[0], cells, 8));
  std::printf("  port box=16 rem=16: %.4e s/pair  nodes/call=%zu  (10 reps)\n",
              runPorted(16, 16, 10),
              rk::box_nodes(16) + rk::remainder_nodes(points[0], cells, 16));
  std::printf("  port box=256 rem=16 (upstream defaults): %.4e s/pair  "
              "nodes/call=%zu  (2 reps)\n",
              runPorted(256, 16, 2),
              rk::box_nodes(256) + rk::remainder_nodes(points[0], cells, 16));
  std::printf("  port box=256 rem=256 (upstream defaults): %.4e s/pair  "
              "nodes/call=%zu  (1 rep)\n",
              runPorted(256, 256, 1),
              rk::box_nodes(256) + rk::remainder_nodes(points[0], cells, 256));
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::runtime_error(
          "usage: refkernel_port <probes|convergence|accuracy FILE|bench>");
    std::string const mode = argv[1];
    if (mode == "probes")
      return probes();
    if (mode == "convergence")
      return convergence();
    if (mode == "accuracy") {
      if (argc < 3)
        throw std::runtime_error("accuracy needs a reference table");
      return accuracy(argv[2]);
    }
    if (mode == "bench")
      return bench();
    throw std::runtime_error("unknown mode " + mode);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "refkernel_port: %s\n", e.what());
    return 2;
  }
}
