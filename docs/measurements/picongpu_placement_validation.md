# picBunch -> PIConGPU placement and particle-data validation

Status: completed on `rosi_v100`. Placement is correct. The requested
`unit_mass` value causes a 1e4 momentum scale error on PIConGPU restart.

## Paths

- picBunch source/build: `/root/picbunch-pic/src`, `/root/picbunch-pic/build/picBunch`
- Test data: `/root/picbunch-meas/bunch.h5`
- Requested config: `/root/picbunch-meas/params.toml`
- picBunch checkpoint: `/root/picbunch-meas/ckpt/checkpoint_000000.h5`
- PIConGPU step-1 checkpoint: `/root/picbunch-meas/out/checkpoint_000001.h5`
- Diagnostic config/run: `params_diag_mass.toml`, `ckpt_diag/`, `out_diag/`
- Logs: `picbunch.log`, `picongpu.log`, `picbunch_diag.log`, `picongpu_diag.log`
- Analysis: `make_bunch.py`, `compare_final.py`, `compare_final.json`, `final_out.txt`

## Build

```
tar xzf latest.tgz -C /root/picbunch-pic/src --strip-components=1
env -u LD_PRELOAD PATH=/usr/local/cuda/bin:$PATH cmake -S /root/picbunch-pic/src -B /root/picbunch-pic/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPICBUNCH_ENABLE_CUDA=ON -DPICBUNCH_ARCH=70 \
  -DopenPMD_DIR=/opt/openpmd-hdf5-0.17.1/lib/cmake/openPMD \
  -DPICBUNCH_BUILD_TESTS=OFF -DPICBUNCH_BUILD_GTESTS=OFF
env -u LD_PRELOAD PATH=/usr/local/cuda/bin:$PATH cmake --build /root/picbunch-pic/build --target picBunch -j 16
```

Target `picBunch` (output of `picBunchApp`). openPMD 0.17.1 HDF5, CUDA 12.6, sm_70.

## Input and placement

`bunch.h5` is an openPMD 2.0.0 groupBased file with 12 marker electrons.
Positions are SI metres (unitSI = 1), momenta are `u = p/(m_e c)` (unitSI =
`m_e c`), charge `-e`, mass `m_e`, weight 1. The offsets are asymmetric
(+x, -x, +y, +z) and the arithmetic mean is removed so the
charge-weighted centroid starts at zero.

```
env -u LD_PRELOAD PATH=/usr/local/cuda/bin:$PATH \
  /root/picbunch-pic/build/picBunch --input /root/picbunch-meas/bunch.h5 \
  --destination /root/picbunch-meas/ckpt --config /root/picbunch-meas/params.toml
```

picBunch reports target centre (1.6e-6, 1.6e-6, 1.6e-6) m and patch
(0,0,0)/(32,64,32). Measured picBunch centroid fraction:
`[0.5000000006, 0.25, 0.5]` of the 32x64x32 domain.
picBunch vs input after the placement shift: max position deviation
`1.99e-15 m`, max `|du| = 2.38e-8` (float32 rounding).

## PIConGPU run

```
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 mpirun --allow-run-as-root -n 1 \
  /home/ubuntu/pr24/w7c-20261102/build/picongpu -d 1 1 1 -g 32 64 32 --periodic 1 1 1 \
  --currentInterpolation none \
  --checkpoint.restart --checkpoint.restart.step 0 \
  --checkpoint.restart.directory /root/picbunch-meas/ckpt \
  --checkpoint.restart.file checkpoint \
  --checkpoint.period 1 --checkpoint.directory /root/picbunch-meas/out \
  --openPMD.pluginConfig /home/ubuntu/pr24/w7c-20261102/deck-static/etc/openPMD_config_E.toml \
  --openPMD.pluginConfig /home/ubuntu/pr24/w7c-20261102/deck-static/etc/openPMD_config_B.toml -s 1
```

Result: exit 0, `100 % = 1`, `full simulation time: 20.649 sec`, wrote
`out/checkpoint_000001.h5`. No extra checkpoint plugin flags were needed; the
deck already writes particle checkpoints with `--checkpoint.period` and
`--checkpoint.directory`.

`-d 1 1 1` is required. Without it PIConGPU aborts with
`expression (devices.size() >= 2 && devices.size() <= 3) failed ... Invalid
number of devices`. That was the only run failure; no checkpoint was written in
that attempt.

## openPMD metadata used

- Input: `openPMD 2.0.0`, `iterationEncoding groupBased`, `basePath /data/%T/`,
  12 particles, `position/*` unitSI 1.0, `momentum/*` unitSI `m_e c`.
- picBunch checkpoint: `openPMD 1.1.0`, `iterationEncoding fileBased`,
  `iterationFormat checkpoint_%06T`, `meshesPath fields/`, `particlesPath
  particles/`. `position/*` and `positionOffset/*` unitSI = 1e-7 (cell size).
  `momentum/*` unitSI = `picongpu.unit_mass * unit_speed` (requested:
  2.7309245307e-22; diagnostic: 2.7309245345e-18). `weighting` unitSI 1,
  `macroWeighted` 1, `weightingPower` 1, `id` uint64.
- PIConGPU step 1: `openPMD 1.1.0`, fileBased, iteration attributes
  `unit_mass 9.1093837139e-27`, `unit_length 5.7735e-8`, `timeUnitSI
  1.9258332015464702e-16`, `position/*` unitSI 9.99999982e-8,
  `momentum/*` unitSI 2.7309245345e-18.

PIConGPU restart ignores particle `unitSI`; `LoadParticleAttributesFromOpenPMD`
assigns the stored value directly to the internal attribute. PIConGPU internal
momentum is in units of `sim.unit.mass() * c`, and `sim.unit.mass()` is
`m_e * 10000` here, so a checkpoint value `v` corresponds to physical
`u = v * 10000.0000136`.

## Comparison (12 markers matched by their unique momentum pattern)

Run A, requested `unit_mass = 9.1093837015e-31`:

- step-1 centroid fraction: `[0.501922, 0.250885, 0.503431]` (x, y, z).
- raw momentum records bitwise equal (`max|diff| = 0`), but physical
  step-1 momentum is `10000.0000136x` the intended value; max relative
  momentum error `9999.0000136`.
- max position change per axis: dx 0.556, dy 0.504, dz 0.504 cells. This equals
  `dt*c*beta` for the scaled momentum (`beta ~ 1`), with the ballistic model
  reproducing the displacement to `5.2e-7` cells. y is not a small drift; every
  marker moves at about c.
- axis consistency: `d_i/u_i` is the same positive scalar on all three axes
  (spread `9.4e-10`), so there is no axis permutation and no mirror.

Run B, diagnostic `unit_mass = 9.109383713899999e-27` (= `sim.unit.mass()`):

- step-1 centroid fraction: `[0.500755, 0.250446, 0.501466]`.
- physical momentum preserved exactly, `max|du| = 0`.
- max position change per axis: dx 0.281, dy 0.210, dz 0.210 cells, matching
  `dt*c*u/gamma` to `4.1e-7` cells.
- axis consistency spread `8.9e-6`, all signs positive: identity axes, no
  mirror.

## Verdict

- picBunch placement: correct. Centroid at x = 0.5, z = 0.5, y = 0.25 of the
  domain, no x/z mirror, no axis permutation. The input offsets survive
  placement to float64/float32 precision.
- Particle records: positions, patches, ids, weights survive the restart
  bitwise or to float32 precision.
- Blocker: with the requested `picongpu.unit_mass = 9.1093837015e-31`, the
  written momentum unit does not match PIConGPU's restart convention
  (`sim.unit.mass() = 9.1093837139e-27`). PIConGPU restarts the particles
  ultra-relativistically. Using `unit_mass = 9.109383713899999e-27` fixes the
  physical momentum and gives the small one-step drift. The trajectory
  direction remains correct in both runs, which rules out a mirror or axis swap.
