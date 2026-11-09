# picBunch checkpoint, 4-rank PIConGPU restart on a decomposable grid

Status: completed on `jureca-h100` (node `jrc0880`, dc-h100, 4x H100 PCIe),
2026-11-08, 21:00 UTC. The 4-rank restart of a picBunch-written openPMD
checkpoint exits 0 and reproduces the checkpoint fields bitwise after one
step, on a geometry whose global grid is a multiple of the per-rank grid in
every dimension.

## Geometry and decomposition

- Global grid: 24 x 24 x 96 cells of 1e-7 m, origin (0, 0, 0).
- Decomposition: 4 ranks in z, `-d 1 1 4`; per-rank grid 24 x 24 x 24, so
  the global size is an integer multiple of the per-rank size on every axis.
  With this ratio PIConGPU does not auto-adjust the grid: no
  `Auto adjust` / `new grid size` lines appear in the restart log.
- The 24 x 24 x 12 grid of the single-rank record
  (`picongpu_placement_validation.md`) is not decomposable by 4 in x or y
  (24/4 = 6, but the y-local size 12 is not a multiple of the supercell
  size 8), and PIConGPU enlarges it to 48 x 48 x 12 before the 4-rank field
  load. That mismatch, not the checkpoint, is what aborts the 24 x 24 x 12
  run: ranks at offset 24 read a 48-chunk from a 24-wide dataset.

## Input and checkpoint

- Species: one electron species `e`, 256 particles, all moving along +y with
  u = 9.949874367 (beta = 0.9949874367), charge -e, mass m_e, weight 1,
  ids 0..255. The compact transverse extent (sigma 0.35 cells in x and y,
  1.0 cell in z, centroid cell (12, 12, 48), RNG seed 20261108) puts every
  particle in z cells 45.7 to 51.2, i.e. in the second and third quarter of
  the 96-cell domain.
- picBunch configuration: domain cells [24, 24, 96], placement
  `patch = "domain"`, `patch_count = 4`, offset_unit metres,
  `precision = "float32"`, `delta_t = 1.9258332015464702e-16`,
  `cell_size = [1.0e-7, 1.0e-7, 1.0e-7]`.
- The writer emits the checkpoint as a single openPMD 1.1.0 fileBased file
  `checkpoint_000000.h5` with one global E/B field record of shape
  (96, 24, 24) (z, y, x), Yee staggering, gridUnitSI 5.7735e-8 m, and the
  electron species as four particle patches (one per output rank,
  `patch_count = 4`). The compact bunch occupies two of the four 24-cell
  z slabs: per-slab particle counts [0, 133, 123, 0] and written
  `particlePatches/numParticles = [0, 218, 38, 0]`,
  `numParticlesOffset = [0, 0, 218, 256]` (one-cell overlap between
  neighbouring z regions). The empty patches are written, not omitted.
- A restart on this PIConGPU build also requires the second, ion species
  `i` to exist in the file even when empty; a checkpoint carrying only `e`
  aborts with `Key 'i' does not exist (read-only)` before any field is
  read. The checkpoint archived here therefore carries an empty `i` species
  (zero particles, matching four-patch metadata) in addition to `e`.
- Command form (both binaries at the revisions below):

  ```
  picBunch --input bunch.h5 --destination ckpt --config ckpt4z.toml
  mpirun.openmpi --allow-run-as-root -n 4 --oversubscribe picongpu \
    -d 1 1 4 -g 24 24 96 --periodic 1 1 1 --currentInterpolation none \
    --checkpoint.restart --checkpoint.restart.backend openPMD \
    --checkpoint.openPMD.ext h5 \
    --checkpoint.restart.directory ckpt --checkpoint.restart.step 0 \
    --checkpoint.period 1 --checkpoint.directory out -s 1
  ```

## Verdict

- Exit status: 0, one step, `full simulation time: 39.320 sec`, no abort,
  no auto-adjust.
- All six field components E/x, E/y, E/z, B/x, B/y, B/z (float32, shape
  (96, 24, 24)) are bitwise equal between the picBunch-written
  `checkpoint_000000.h5` and the PIConGPU-written `checkpoint_000001.h5`
  after one step: `bitwise_equal = true`, `max_abs_diff = 0.0` for every
  component. The comparison is a global read of both files with h5py,
  independent of the MPI layout.
- The per-slab particle distribution [0, 133, 123, 0] above is the
  distribution of this compact bunch; it is not a reproduction of any other
  recorded bunch, and no other per-patch count is claimed.

## Host, versions, artifact hashes

- Host: `jrc0880.jureca` (jureca-h100, partition dc-h100, 4x H100 PCIe,
  144 cpus, CUDA 13.4.92, GCC 13.3.0, OpenMPI 4.1.6).
- picBunch binary: `block1` build `d6c964ae775dbc2f9c23cc66b81e201388284392fe54c09a3b271de48f4f8275`
  (sm_90, ADIOS2 2.12.1, openPMD 0.17.1), run as the documented
  `--input/--destination/--config` interface.
- PIConGPU binary: PIConGPU 0.9.0-dev Release (scratch copy of the
  `pr24w126` deck source, HEAD `b87c053`), sha256
  `f316a1075c637d2eb67c6a3106ffe1bd8f846905b17c7bc088e3b80dbb06dcb8`,
  sm_90, ADIOS2 2.12.1, openPMD 0.17.1, CUDA 13.4.92. Deck units logged in
  the restart: `sim.unit.time 1.92583e-16`, `length 5.7735e-08`,
  `mass 9.10938e-27`, `charge 1.60218e-15`, `eField 8.85076e12`,
  `bField 29523`, `speed 2.99792e8`.
- Analysis: h5py 3.10.0, NumPy 1.26.4.
- Artifacts (node paths under `/home/ubuntu/block1-deck/work/v6b3/`,
  mirrored in the external `v6b3-arch` archive):

  | File | SHA256 |
  |---|---|
  | `bunch_input.h5` | `0ca47bd8d383de25c53496c75a4a0f2d2f2629c9d8bfeb9ea2db7f5a0c51be85` |
  | `picbunch_config.toml` | `5e1addc72f5ab3081d1a63bdcc19fc59c65da384971ae053cb400e3452c82db8` |
  | `ckpt/checkpoint_000000.h5` | `0a16eae4a83ca543ac0e7ff6085e562691825645fa44f6616e6bb1ad4f68ca09` |
  | `out/checkpoint_000001.h5` | `f1a76c6a9a5b05fe40998b00eb4dca27a5f7217fd339cb12159478f0761ad483` |
  | `restart4.log` | `5c2a4b3f3a81ac9800f1d57d1422b3e12bec2f7be8579d58f3a602f37c5dbcb9` |
  | `picbunch.log` | `31b91ed17edde09774077dec188ea97b7a5d372ca36b0439d0ec81034950ab1d` |

The input generator, the picBunch configuration, both checkpoints and both
logs are archived with this record; the per-component field verdict is
recomputed from the two archived HDF5 files, not from the log.
