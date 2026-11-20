# Chunked field solve: the distributed algorithm with a staged exchange

The production target is a `640 x 1536 x 640` output grid at FFT padding 2.  The
padded box is `1280 x 3072 x 1280` (`5.03e9` cells).  This path does **not**
introduce a new global FFT.  It runs the existing distributed algorithm, the
one the four-rank MPI path uses, with the collective exchange replaced by a
staging file.

## The distributed algorithm being reused

`DistributedFft3D` splits the padded box into contiguous z-slabs.  Each rank
holds all x and y nodes for its z planes (`Mx*My*nz` complex values).  It runs
the x and y line transforms on that slab, then transposes so that each rank
holds an x-slab of the spectrum (all z and y for its x range).  The z transform,
the kernel multiply and the inverse z transform are applied in that layout;
a second transpose returns to the z-slab and the inverse y/x transforms finish
the transform.  The index arithmetic of both transposes is in
`DistributedFft.hpp` (`packForward`, `unpackForward`, `packBackward`,
`unpackBackward`, `buildTransposeCounts`).

The chunked path reuses that class unchanged for the slab split, the plans, the
local transforms and the pack/unpack arithmetic.  One process plays every
virtual rank in turn.  The only replaced operation is the `MPI_Alltoallv` call:
each source rank writes its packed slice to a staging file at the offset of the
all-to-all matrix, and each destination reads its column from the same file.
A virtual rank and a real rank therefore execute the same operations on the same
bytes, which is why the two agree to round-off (and, for the test case, bit for
bit).

## Exchange layout and the three passes

For one y the all-to-all matrix is `[source z-rank][destination x-rank]`.  The
block `(s,d)` has `nz_s * nx_d` complex values and starts at

```
forward:  y*(Mx*Mz) + sum_{s'<s} nz_s'*Mx + nz_s*x0_d
backward: y*(Mx*Mz) + sum_{r'<r} nx_r'*Mz + nx_r*z0_s
```

in complex elements.  Those are exactly the displacements `buildTransposeCounts`
already computes, so the staged bytes and the MPI message of a pair are the
same.  The file slice per y holds `Mx*Mz` complex values; the whole file holds
`Vc = Mx*My*Mz*16` bytes.

The exchange is split into three passes so that only one virtual rank's buffers
are alive:

1. every z-slab rank runs the local x/y transforms and writes its packed slice
   for every y;
2. every x-slab rank reads its column, unpacks, runs the z transform, and
   writes the source spectrum;
3. per component, every x-slab rank reads the source spectrum and the kernel
   column, z-transforms the kernel, multiplies, inverse-z-transforms and writes
   the backward slices; every z-slab rank then reads its backward column,
   unpacks, runs the inverse y/x transforms and emits the output z-slabs.

## Peak memory

With `N` virtual ranks, `nz = ceil(Mz/N)` and `nx = ceil(Mx/N)`:

```
peak = 16 * ( Mx*My*nz            // one z-slab working array
            + 2*nx*My*Mz          // source spectrum + kernel/product spectrum
            + 2*max(Mx*nz, Mz*nx) // one forward and one backward packed slice
            )
```

plus the FFT plans and the writer's in-flight slabs.  It has no term that grows
with the padded volume once `N` is fixed.

| N | nz | nx | peak (production) |
| --- | --- | --- | --- |
| 6 | 214 | 214 | ~30 GB |
| 22 | 59 | 59 | ~8.4 GB |
| 32 | 40 | 40 | ~7.6 GB |
| 64 | 20 | 20 | ~3.8 GB |
| 1280 | 1 | 1 | ~189 MB |

The slab size is selected by `--chunk-cells`: the solver uses
`ceil(chunk-cells / (Mx*My))` planes per virtual rank.  `--virtual-ranks N`
overrides the derived count.

## I/O volume

`Vc = Mx*My*Mz*16` is one complex padded volume.  The three passes move

```
source:       write Vc (forward exchange) + read Vc + write Vc (source spectrum)  = 3 Vc
per component: write Vc (kernel) + read Vc (source spectrum) + read Vc (kernel)
               + write Vc (backward) + read Vc (backward)                          = 5 Vc
six components:                                                                    30 Vc
total:                                                                             33 Vc
```

independent of the virtual rank count.  At padding 2, `Vc = 2*Vreal`, so
`33 Vc = 66 *` the real padded volume.

## Measured results

### Start small: 64x96x64, 2000 particles

The first end-to-end case is a `64x96x64` grid at padding 2 with 2000
particles; a four-chunk cycle completes in seconds.  Peak RSS is
`/usr/bin/time -v`.

| path | ranks | wall | peak RSS | output agreement |
| --- | --- | --- | --- | --- |
| in-core p2 | 1 | 2.07 s | 141.6 MB | reference |
| MPI distributed p2 | 4 real | 2.9 s | - | reference |
| staged p2, N=4 | 4 virtual | 4.00 s | 96.5 MB | bit-identical to MPI 4 (0.0) |
| staged p2, N=4 | 4 virtual | 4.00 s | 96.5 MB | 3.1e-16 vs in-core |

The virtual-versus-real-rank comparison is the strongest check: the staged
path and the four-rank MPI path produce the same six components bit for bit.

### Chunk-count independence of peak RSS

Same 64x96x64 case, `--chunk-cells 4096`:

| N | nz | nx | wall | peak RSS | peak chunk (formula) |
| --- | --- | --- | --- | --- | --- |
| 2 | 64 | 64 | 3.15 s | 136.4 MB | 0.076 GB |
| 4 | 32 | 32 | 4.00 s | 96.5 MB | 0.038 GB |
| 128 | 1 | 1 | 75.8 s | 56.6 MB | 0.001 GB |

Peak RSS does not grow with the chunk count; it falls towards the fixed runtime
baseline (~55 MB) as the formula shrinks.  The stage never holds the volume:
each phase writes its output to a staging file and the next phase reads only
its column.  The large-N wall time shows the exchange granularity cost (the
per-y block becomes tiny); N between 4 and 22 is the practical range.

### Density before any transform

`ChunkedDensity.SlabsEqualInCoreDensityBitForBit` deposits the same 64x96x64
cloud slab by slab and compares the concatenation with
`depositTSCPaddedHost`: **0 differing nodes** over `128*192*128` padded nodes.
`ChunkedDensity.BoundaryParticleWeightAppearsExactlyOnce` puts one particle at
`z=32.0e-6`, exactly on the boundary of a four-chunk split.  Its TSC support
reaches z=31 (chunk 0) and z=32,33 (chunk 1); both chunks receive mass, their
partial sums add to the analytic `weight*charge`, and the full concatenation is
bit-identical to the in-core density.  Measured deposited sum
`2.0027207924999999e-19` versus analytic `2.0027207924999999e-19`, relative
error 0.  There is no double counting and no loss.

### Near field

The chunked path does **not** enable the near shell (it rejects
`--near-shell`).  The in-core near shell computes the mesh subtraction once
over the whole domain and the exact correction per tile.  Recomputing that
whole-domain correction per chunk would be redundant: the sources are
replicated and the exact correction at a chunk depends only on sources within
`radius + 1` cell, so a chunked implementation must use a per-chunk source list
restricted to that neighbourhood (or one streaming whole-domain pass) rather
than repeat the global correction N times.  Measured on the 64x96x64 case, the
in-core `--near-shell fast` correction costs 2.33 s on top of the 2.07 s base
(4.40 s total, 2.17e6 exact evaluations, 158 MB RSS); `--near-shell exact` did
not finish in five minutes on this host.  A per-chunk restricted list would pay
that exact-evaluation cost only for the local sources; the rejected alternative
would multiply the 2.33 s by N.  The decision is therefore per chunk with a
restricted source list, once that path is implemented; no per-chunk
recomputation exists today because the mode is not exposed.

Host: `fwk394` (Intel Xeon W-2175, 28 threads, 125 GiB RAM), CPU backend,
Release, FFTW 3.3.10, openPMD 0.17.1 HDF5, staging on `/home/agent`.  Peak RSS
is `/usr/bin/time -v`; I/O is the counted staging traffic.

| grid | path | N | chunk | wall | peak RSS | staging read | staging write | peak staging | peak chunk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16^3 | in-core p2 | - | - | 0.35 s | 34 MB | - | - | - | - |
| 16^3 | staged p2 | 8 | 4096 | 1.3 s | 34 MB | - | - | - | - |
| 192x288x192 | in-core p2 | - | - | 28.2 s | 3.11 GB | - | - | - | - |
| 192x288x192 | in-core p1 | - | - | 12.6 s | 1.37 GB | - | - | - | - |
| 192x288x192 | staged p2 | 22 | 4 194 304 | 83.1 s | 623 MB | 25.82 GB | 19.03 GB | 4.08 GB | 0.191 GB |
| 192x288x192 | staged p2 | 6 | 16 777 216 | 81.4 s | 1003 MB | 25.82 GB | 19.03 GB | 4.08 GB | 0.680 GB |
| 256x384x256 | staged p2 | 32 | 4 194 304 | 193.5 s | 722.6 MB | 61.20 GB | 45.10 GB | 9.66 GB | 0.302 GB |

The grid column identifies each row: 16^3 is the smoke case, 192x288x192 is
the scaling case, and 256x384x256 is the largest validated run.  The 256 grid
has `Vc = 3.22 GB`, so the model predicts `33 Vc = 106.3 GB`; the measured
61.20 GB read plus 45.10 GB write totals 106.30 GB.  Peak staging is
`3 Vc = 9.66 GB`.  At 256x384x256 the near-field agreement against in-core
padding 1 is `4.89e-16` and the whole-domain difference from the periodic tail
is `1.51e-3`.

The 192 grid has `Vc = 1.36 GB`, so the model predicts `33 Vc = 44.9 GB`; the
measured 44.85 GB matches.  Peak staging is `3 Vc = 4.08 GB`, also exact.  The
two chunk sizes move identical bytes; the larger chunk uses 61% more RSS and
the same wall time.

Agreement at 192x288x192: staged p2 versus in-core p2 `3.4e-16` (FFT round-off;
the two plans batch differently), staged p2 versus in-core p1 in a 25^3-cell
near-field box `7.6e-16`, whole-domain p1 difference `2.8e-3` (the expected
periodic tail).  The 16^3 staged p2 is bit-identical to in-core p2.

Virtual versus real ranks: 16^3 with 4 virtual ranks and 4 real MPI ranks is
bit-identical on all six components.  This is the committed
`picBunch.ChunkedVirtualVsMpi` CTest.

## Production grid

`Vc = 80.5 GB`, `33 Vc = 2.66 TB`, peak staging `3 Vc = 241 GB`.  With N=32 the
design peak is ~7.6 GB; with N=64, ~3.8 GB; the wall time scales with the
measured 192-grid 83 s by the data ratio (59x), or about **80 min**, and the
single-threaded FFT is the other half of that.  `fwk394` has about 60 GB free,
below the 241 GB staging high-water mark, so the production run cannot finish
on this node's scratch.  It needs a scratch with at least 241 GB free; the
memory ceiling is the N-dependent formula above, not the padded volume.

## Switch

```
--solve-mode auto|chunked|none|in-core [--chunk-cells <N>]
            [--virtual-ranks <N>] [--staging-dir <dir>] [--keep-staging]
```

`--solve-mode` accepts exactly `auto`, `chunked`, `none` and `in-core`; any
other spelling is rejected with exit 2.  `none` and `in-core` select the
whole-grid in-core solve, the solver that keeps the padded volume in memory;
`none` is the canonical spelling for "never use the chunked solver".

`auto` is the default.  It decides once per run, before grid construction and
before the first allocation.  It keeps the in-core solve when the estimated
requirement fits the per-rank budget, and switches to the staged solve when
that solver is available in the build and its own estimate fits.  The switch
prints exactly one line on rank 0:

```
picBunch: solve mode: auto -> chunked (required <X> GiB, available <Y> GiB per rank, reason <key>)
```

When both modes are infeasible the run refuses, with a message that names the
required bytes, the available bytes per rank and the available alternatives.
When the capacity is unknown `auto` keeps the historical in-core behaviour.
`auto` never falls back to a mode the build cannot run.

`chunked` forces the staged path and is not gated by the in-core estimate; a
large chunked run no longer fails with an in-core estimate for a solver it
will not use.

### Multi-rank behaviour

`chunked` runs on any rank count.  Each MPI rank executes the global staged
solve, and only the rank that owns a range of global output z-planes writes
them, through the existing collective routing.  The written field is the
exact global transform and matches the distributed in-core reference; it was
verified bitwise equal on divisible and non-divisible grids, at padding 1 and
2, with 1, 3 and 4 ranks.  The cost is I/O: each rank performs the full
staged exchange, so the staging traffic multiplies with the rank count, and
the FFT work is not parallelised by MPI.  `chunked` is a memory-saving mode,
not a speed mode.

The staging directory is per rank: `<staging-dir>/rank-<r>` holds rank `r`'s
exchange files.  `--keep-staging` keeps them for inspection.

The staged solve is host/CPU-only at runtime.  On a CUDA or HIP build `auto`
never selects it, and an explicit `chunked` fails with a message that says so;
the alternative on a GPU build is `--fft-mode distributed`.  Making the
staged solve device-memory-bounded on GPUs is future work.

The staged solve does not combine with `--near-shell`, `--velocity-moments` or
`--global-grid`.  On the direct `--input`/`--output` path it additionally
requires an explicit `--grid`, `--origin` and `--cell-size`, a single velocity
box, and `--output` when there is no checkpoint.  With `--output` the
single-rank run stores every component z-slab through `ChunkedFieldWriter`
into one series whose datasets have the full global extent from the first
`storeChunk`; that streaming writer path is unchanged.  With
`--restart-format picongpu` the field is accumulated for the existing
checkpoint writer.

### The config path

The `--config` path honours `--solve-mode`, `--chunk-cells`,
`--virtual-ranks`, `--staging-dir`, `--keep-staging`,
`--deterministic-deposit`, `--shape`, `--input-iteration` and
`--output-iteration`; previously these were accepted and silently ignored,
and `chunked` was rejected outright.  The optional TOML key

```toml
[solve]
solve_mode = "auto"   # auto|chunked|none|in-core
```

selects the same policy, with an explicit command-line value taking
precedence.  The config path writes one collective checkpoint through the
existing openPMD checkpoint writer; each rank stores only its own slab.
