# Near-slice plane errors, fresh valid3 runs (v0.9.2)

Case ids `nearslice_g1` and `nearslice_g10` of the
`picbunch-repro-20261108` validation pass (block 1, pass `valid3`,
2026-11-08 20:18--21:44 UTC). Host `jrc0880.jureca` (4x H100 PCIe 80 GB,
exclusive `dc-h100` allocation); `nearslice_g10` ran alone on GPU1
(`CUDA_VISIBLE_DEVICES=1`, 81,072 MiB free before and after, peak observed
118 MiB, host RSS ~82 MB). Backend ADIOS2-BP5
(`OPENPMD_BACKEND_CONFIG={"backend":"adios2"}`) with openPMD-api 0.17.1 and
ADIOS2 2.12.1; CUDA 13.4.92, gcc 13.3.0.

Code: picBunch source `478f6e1b7fea4c69dc0e964512862f33121df04c` (v0.9.2)
plus the ADIOS2 2.12.1/BP5 build fix; binary sha256
`cdbcd8e64ed206af3399a607802d3e4f4a74e977f57bd4afe1edbe4e75db6da1`. The
values quoted in Appendix `app:slices` come from these v0.9.2 runs; the
old high ends `0.1067` (gamma=1) and `0.5704` (gamma=10) are the same
z-y `E_z` plane's last domain row (`z=15`, worst pixel `(z=15, y=17)`)
from the v0.4.0 standalone driver (`pr24-20261102/SL3/data/plane_stats.csv`,
v0.4.0 = `bc0037c5fa0ddb8e2b67c0a75815cda0349a7b27`) and are superseded
by `2.46e-05` and `0.0954` from the v0.9.2 runs.

## Geometry (both cases, "paper geometry")

- grid `32 x 32 x 16` nodes, cell `2e-7` m, origin `0,0,0`
- TSC shape, tolerance `1e-3`, `--velocity-bins 1`
- near shell `--near-shell exact`, `R=3` (gamma=1) / `R=4` (gamma=10)
  mesh nodes of the `h=1` grid
- planes: x-y at `z=3.2e-6` m (`iz=8`), z-y at `x=6.4e-6` m (`ix=16`)
- metric: worst per-component relative error per plane,
  `max|mesh-ref| / plane peak`, unmasked (same convention as the
  mid-slice table)

## nearslice_g1 (R=3)

Command:

```
picBunch --input in/gamma1_1k.h5 --cell-size 2e-7 --grid 32,32,16 \
  --origin 0,0,0 --velocity-bins 1 --tolerance 1e-3 --shape tsc \
  --near-shell exact --near-shell-radius 3 --output out/nearslice_g1.bp
```

- input: `in/gamma1_1k.h5`
  sha256 `9b2b8b88996f45db5d0c14aa1e42a0a8c133ada599ef578e93727efef5796ba0`
  (committed `refs/bunches/gamma1-1k/bunch.h5` physical npz, u=0, rebuilt
  to openPMD)
- reference: `refs/mid_g1_1k_32.bin`
  sha256 `a6f51a13c2c2c61b9022fe4fc2378bb4602c683857b2f242d00730a00801eb63`
  (committed `refs/reffield/gamma1_1k_reference.h5`, order-4 driver,
  max level 5, converted)
- output: `out/nearslice_g1.bp` `data.0`
  sha256 `8504349508b9100ec84652b8c35b1e8c7861a700ecdd196fb0672e9376b5dbc2`;
  `out/nearslice_g1.h5`
  sha256 `148f1141794cfa94481e3da216c2cdc288d5730d78227d1afcd5e7ead3107662`
- exact shell: `4,245,120` (source, node, component) evaluations; wall
  156 s (02:20:04Z to 02:22:40Z run window)

| plane | Ex | Ey | Ez | By | Bz |
|---|---|---|---|---|---|
| x-y (`z=8`) | 1.706e-3 | 2.093e-3 | 5.462e-3 | 0 | 0 |
| z-y (`x=16`) | 8.972e-3 | 2.445e-3 | 1.943e-3 | 0 | 0 |

Per-component plane worst range: `1.71e-3` (x-y `E_x`) to `8.97e-3`
(z-y `E_x`); `B` identically zero for the stationary bunch. The
v0.4.0-driver high end `0.106685` (z-y `E_z`, worst pixel `(z=15,y=17)`,
the last domain row) is `2.4622e-05` on that same row in the v0.9.2
output; worst elsewhere `1.943e-3`.

## nearslice_g10 (R=4)

Command:

```
CUDA_VISIBLE_DEVICES=1 OPENPMD_BACKEND_CONFIG='{"backend":"adios2"}' \
picBunch --input in/gamma10_1k.h5 --cell-size 2e-7 --grid 32,32,16 \
  --origin 0,0,0 --velocity-bins 1 --tolerance 1e-3 --shape tsc \
  --near-shell exact --near-shell-radius 4 --output out/nearslice_g10.bp
```

- input: `in/gamma10_1k.h5`
  sha256 `236370430b877abab563ca47407c1639e7078d823adaf41245e84ab70ff9369f`
  (committed `refs/bunches/gamma10-1k/bunch.h5` physical npz,
  u=(9.94987,0,0), i.e. beam axis `+x`, rebuilt to openPMD)
- reference: `refs/mid_g10_1k_32.bin`
  sha256 `9b075e64f0d5130859ad0d0f39264f9106c690dd4296ab8babaa7e6015f6759f`
  (committed `refs/reffield/gamma10_1k_reference.h5`, order-4 driver,
  max level 5, converted)
- output: `out/nearslice_g10.bp` `data.0`
  sha256 `ac4a226d595f16df99d6490b33abf411d42d01badc1718daacc8aed9c7d7ffad`;
  `out/nearslice_g10.h5`
  sha256 `1adb951096d457e1ba82badc2cbc6af25df73e55f6bf82de9c5d2964039b6b6c`
- exact shell: `7,460,736` (source, node, component) evaluations; near-shell
  wall `1,962,039.6` ms, compute wall `1,962,956.6` ms (32 m 43 s), run
  21:10:52Z to 21:43:39Z UTC
- field maxima: `max|E| = 1.679e6` V/m, `max|B| = 4.975e-3` T

| plane | Ex | Ey | Ez | By | Bz |
|---|---|---|---|---|---|
| x-y (`z=8`) | 7.434e-2 | 1.246e-1 | 6.728e-2 | 7.060e-2 | 1.181e-1 |
| z-y (`x=16`) | 1.826e-1 | 1.231e-1 | 9.544e-2 | 8.774e-2 | 9.109e-2 |

Per-component plane worst range: `6.73e-2` (x-y `E_z`) to `1.83e-1`
(z-y `E_x`); `B_x` identically zero for motion along `+x`. The
v0.4.0-driver worst `0.570423` (z-y `E_z`, same `(z=15,y=17)` edge row)
is `9.544e-2` in the v0.9.2 output.

## Superseded v0.4.0-driver rows (for the transparency statement)

From `pr24-20261102/SL3/data/plane_stats.csv` (standalone driver, v0.4.0
= `bc0037c5fa0ddb8e2b67c0a75815cda0349a7b27`, host `rosi_v100`), near-mode
worst per component, same geometry and planes:

| case | plane/component | v0.4.0 driver | v0.9.2 (this record) |
|---|---|---|---|
| g1, R=3 | z-y Ez | 0.106685 | 1.943e-3 (row z=15: 2.462e-05) |
| g10, R=4 | z-y Ez | 0.570423 | 9.544e-2 |

Both v0.4.0 worst pixels are the last domain row `z=15`, pixel `(z=15,y=17)`;
same pair counts (`4,245,120` / `7,460,736`), so the difference is driver
version at the domain edge, not a missing plane or radius.
