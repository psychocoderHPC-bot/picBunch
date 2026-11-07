# picBunch H100 measurement campaign

Generated on jureca-h100. Medians over 3 repeats. All runs serialized on GPU 0.

## Main cases (N, box cells, 1 rank, 1 velocity bin)

| case | N | box | deposit | kernel | fft | transpose | readout | writer | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g100_N10000 | 10000 | 100 | 3.971 | 3.965 | 33.496 | 0.000 | 24.910 | 70.410 | 134.094 |
| g100_N100000 | 100000 | 100 | 5.367 | 4.006 | 32.878 | 0.000 | 28.153 | 58.763 | 128.366 |
| g100_N1000000 | 1000000 | 100 | 20.735 | 4.010 | 32.447 | 0.000 | 26.206 | 53.087 | 139.036 |
| g100_N2000000 | 2000000 | 100 | 38.908 | 3.942 | 31.284 | 0.000 | 21.390 | 53.777 | 147.899 |
| g20_N10000 | 10000 | 20 | 1.389 | 0.972 | 1.266 | 0.000 | 21.464 | 4.429 | 29.429 |
| g20_N100000 | 100000 | 20 | 2.942 | 0.759 | 1.179 | 0.000 | 22.123 | 5.113 | 32.373 |
| g20_N1000000 | 1000000 | 20 | 17.648 | 0.620 | 1.312 | 0.000 | 17.872 | 4.938 | 42.621 |
| g20_N2000000 | 2000000 | 20 | 36.089 | 0.642 | 1.531 | 0.000 | 17.522 | 5.107 | 60.507 |

## Multi-velocity (N=200, 5% energy spread, 5 deg angular spread)

| case | bins | deposit | kernel | fft | readout | writer | total | maxE (V/m) | maxB (T) | max|Bz| (T) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mv_b16 | 16 | 7.638 | 4.078 | 14.303 | 60.852 | 4.546 | 92.191 | 28030.8 | 1.03632e-05 | 4.32504e-06 |
| mv_b64 | 64 | 10.283 | 6.236 | 25.870 | 54.096 | 4.921 | 100.641 | 29052.7 | 1.01313e-05 | 4.12569e-06 |

## MPI scaling (N=2e6, velocity bins 1)

| case | grid | ranks | deposit | kernel | fft | transpose | readout | writer | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi_g128_r1 | 128 | 1 | 42.238 | 7.609 | 30.816 | 0.000 | 34.263 | 413.931 | 528.906 |
| mpi_g128_r2 | 128 | 2 | 43.158 | 6.521 | 22.235 | 0.000 | 17.441 | 412.226 | 501.542 |
| mpi_g128_r4 | 128 | 4 | 43.998 | 6.152 | 22.441 | 0.000 | 22.799 | 416.254 | 511.602 |
| mpi_g256_r1 | 256 | 1 | 85.533 | 51.411 | 239.949 | 0.000 | 204.175 | 3318.808 | 3899.047 |
| mpi_g256_r2 | 256 | 2 | 87.540 | 42.662 | 175.337 | 0.000 | 105.527 | 3326.889 | 3737.188 |
| mpi_g256_r4 | 256 | 4 | 87.335 | 39.939 | 153.570 | 0.000 | 44.204 | 3323.144 | 3663.864 |

## Exact commands

Main case:

```
/root/picbunch-latest/build/cuda-mpi/picBunch --input /root/meas/input_N<N>.h5 --cell-size 1e-6 --grid <20|100> --origin 0,0,0 \
  --velocity-bins 1 --tolerance 1e-3 --shape tsc --output /root/meas/out.h5
```

Multi-velocity:

```
/root/picbunch-latest/build/cuda-mpi/picBunch --input /root/meas/input_mv_N200.h5 --cell-size 1e-6 --grid 20 --origin 0,0,0 \
  --velocity-bins <16|64> --tolerance 1e-3 --shape tsc --output /root/meas/out.h5
```

MPI:

```
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 /usr/bin/mpirun.openmpi -np <1|2|4> \
  /root/meas/run_mpi.sh /root/picbunch-latest/build/cuda-mpi/picBunch --input /root/meas/input_N2000000.h5 --cell-size 1e-6 \
  --grid <128|256> --origin 0,0,0 --velocity-bins 1 --tolerance 1e-3 --shape tsc --output /root/meas/out.h5
```

`run_mpi.sh` exports `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK` before exec.

Note: the binary prints `reader 0.000` in every stage-timing block (the print path shows the reader clock as zero); reader time is excluded from `total` as printed.
