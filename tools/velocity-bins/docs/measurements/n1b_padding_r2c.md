# N1b results - picBunch mesh convolution measurements (CPU)

Host: rosi_v100, CPU only (no GPU used), OMP_NUM_THREADS=16
python 3.12.3, numpy 2.5.4, scipy 1.18.1
Reference: ~/pr24/pr24w53-ref/indfield.py
Script: /home/ubuntu/pr24/n1-20261102/n1_measure.py

## 1. Wraparound control (vs the direct non-cyclic sum)
Box n=(20,20,20), h=1. Kernel = exact sampled point kernel (matched to the TSC deposit). Error = per-component max|delta| / max|ref_c| over the 20^3 read-out nodes; component-wise, never pooled.  The direct reference is a true linear convolution (method=fft, verified against method=direct).
Reference self-check (8^3, method=fft vs method=direct): max abs diff 9.992e-16
### 1a. padding sweep (worst over the six components)
| case | pad | M | worst comp | worst rel err |
|---|---|---|---|---|
| single_stationary | 1.0x | 20^3 | Ey | 9.275e-03 |
| single_stationary | 1.5x | 30^3 | Ez | 1.905e-16 |
| single_stationary | 2.0x | 40^3 | Ez | 3.811e-16 |
| single_stationary | 3.0x | 60^3 | Ez | 1.905e-16 |
| single_stationary | 4.0x | 80^3 | Ez | 2.858e-16 |
| single_gamma10_oblique | 1.0x | 20^3 | Ey | 6.235e-02 |
| single_gamma10_oblique | 1.5x | 30^3 | Ey | 2.863e-16 |
| single_gamma10_oblique | 2.0x | 40^3 | Ez | 5.638e-16 |
| single_gamma10_oblique | 3.0x | 60^3 | Ez | 2.819e-16 |
| single_gamma10_oblique | 4.0x | 80^3 | Ex | 5.119e-16 |
| cloud100_stationary | 1.0x | 20^3 | Ez | 1.337e-01 |
| cloud100_stationary | 1.5x | 30^3 | Ey | 5.696e-04 |
| cloud100_stationary | 2.0x | 40^3 | Ez | 4.848e-16 |
| cloud100_stationary | 3.0x | 60^3 | Ez | 3.232e-16 |
| cloud100_stationary | 4.0x | 80^3 | Ex | 5.594e-16 |
| cloud100_gamma10 | 1.0x | 20^3 | Bx | 7.529e-01 |
| cloud100_gamma10 | 1.5x | 30^3 | Bx | 6.909e-03 |
| cloud100_gamma10 | 2.0x | 40^3 | Bx | 8.889e-16 |
| cloud100_gamma10 | 3.0x | 60^3 | Ez | 5.517e-16 |
| cloud100_gamma10 | 4.0x | 80^3 | Ez | 6.896e-16 |

Closest to the documented 8.947e-3: single_stationary at pad 1.0x -> 9.275e-03 (Ey)
### 1b. per-component detail of that 1x artefact
| comp | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| rel err | 5.840e-03 | 9.275e-03 | 3.391e-03 | 0 | 0 | 0 |
### 1c. per-component worst error for every pad
case single_stationary
| pad | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| 1.0x | 5.840e-03 | 9.275e-03 | 3.391e-03 | 0 | 0 | 0 |
| 1.5x | 1.718e-16 | 1.709e-16 | 1.905e-16 | 0 | 0 | 0 |
| 2.0x | 1.718e-16 | 3.418e-16 | 3.811e-16 | 0 | 0 | 0 |
| 3.0x | 1.718e-16 | 1.709e-16 | 1.905e-16 | 0 | 0 | 0 |
| 4.0x | 1.718e-16 | 2.563e-16 | 2.858e-16 | 0 | 0 | 0 |

case single_gamma10_oblique
| pad | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| 1.0x | 5.977e-02 | 6.235e-02 | 5.067e-02 | 1.480e-02 | 4.029e-03 | 1.208e-02 |
| 1.5x | 2.559e-16 | 2.863e-16 | 2.819e-16 | 2.172e-16 | 1.900e-16 | 2.701e-16 |
| 2.0x | 5.119e-16 | 4.772e-16 | 5.638e-16 | 5.430e-16 | 3.799e-16 | 5.402e-16 |
| 3.0x | 1.706e-16 | 1.909e-16 | 2.819e-16 | 2.172e-16 | 1.900e-16 | 1.801e-16 |
| 4.0x | 5.119e-16 | 3.818e-16 | 4.228e-16 | 1.086e-16 | 1.900e-16 | 3.601e-16 |

case cloud100_stationary
| pad | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| 1.0x | 9.856e-02 | 1.183e-01 | 1.337e-01 | 0 | 0 | 0 |
| 1.5x | 1.585e-04 | 5.696e-04 | 2.985e-05 | 0 | 0 | 0 |
| 2.0x | 4.195e-16 | 4.397e-16 | 4.848e-16 | 0 | 0 | 0 |
| 3.0x | 2.797e-16 | 2.931e-16 | 3.232e-16 | 0 | 0 | 0 |
| 4.0x | 5.594e-16 | 4.397e-16 | 4.848e-16 | 0 | 0 | 0 |

case cloud100_gamma10
| pad | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| 1.0x | 5.483e-02 | 7.005e-02 | 7.836e-02 | 7.529e-01 | 7.086e-01 | 0 |
| 1.5x | 1.555e-04 | 6.375e-04 | 5.559e-04 | 6.909e-03 | 2.092e-03 | 0 |
| 2.0x | 3.637e-16 | 5.195e-16 | 5.517e-16 | 8.889e-16 | 7.682e-16 | 0 |
| 3.0x | 2.910e-16 | 1.732e-16 | 5.517e-16 | 3.810e-16 | 5.121e-16 | 0 |
| 4.0x | 2.910e-16 | 3.463e-16 | 6.896e-16 | 3.810e-16 | 5.121e-16 | 0 |


## 2. R2C/C2R vs C2C equivalence
Case: cloud100 gamma10, box 20^3, padding 2x (M=40^3), h=1.
max |C2C - R2C| over all six components and all 8000 read-out nodes: 1.665e-16
| comp | Ex | Ey | Ez | Bx | By | Bz |
|---|---|---|---|---|---|---|
| worst rel diff | 1.455e-16 | 2.597e-16 | 4.137e-16 | 3.810e-16 | 4.481e-16 | 0 |
| max |C2C| | 7.631e-01 | 6.412e-01 | 3.145e-03 | 5.464e-02 | 5.420e-02 | 0.000e+00 |
wall: C2C 0.049 s (13 full complex transforms), R2C 0.034 s
Bin-rule check (By): max |full-spectrum product - Hermitian-mirrored half-spectrum product| = 3.580e-15
Self-conjugate bins (DC and the 7 even/even/even Nyquist corners): max |Im| = 2.622e-17
kx-Nyquist plane max |Im| = 1.299e-01 (NOT zero: that plane is Hermitian paired with the ky mirror, only its corners are real)
Rule: store deposit and each kernel as real arrays; take rfftn over the last axis and multiply the two half spectra directly.  No manual conjugation and no factor 2: the half spectrum is closed under the product because G(kx,ky,-kz)=conj(G(-kx,-ky,kz)) for a real kernel, so the kz<0 half is never needed.  Which bins must be exactly real: DC (0,0,0) and the self-conjugate Nyquist corners k=-k mod M, i.e. the 8 bins with each component 0 or M_a/2 for even M_a.  The kz=0 and kz=Mz/2 planes are Hermitian in (kx,ky) with G(-kx,-ky)=conj(G), and the kx=Mx/2 plane is Hermitian in (ky,kz); those planes are NOT real, only their corners are.  Enforce Im=0 on the self-conjugate bins only if the spectrum is assembled analytically; a real kernel already satisfies it.  irfftn must receive the full shape s=M.  Normalisation: divide by M^3 (the scipy inverse already does; a raw vendor C2R does not), and never apply an extra factor 2 on non-DC bins.

## 3. Padding vs distance (vs the independent reference)
Reference: ~/pr24/pr24w53-ref/indfield.py quadrature (tensor-product Gauss-Legendre over the three TSC knot intervals), GL order 16, vectorized over particles after matching indfield to 1e-12.  Padding is relative to the 20-cell source box: M = pad*20.  Observation point at d cells outside the TSC support edge (edge = max particle coordinate + 1.5), at the bunch centre transversally.  The periodic field is read at index p mod M, exactly the periodic-image-lattice answer, and compared with the free-space reference at the physical distance p.  Relative error per component uses the local reference magnitude with a 1e-3*max|ref| floor at that point.
### single_g100_axial
| d (cells) | pad | comp worst | rel err | abs err | ref value | per-component rel (Ex,Ey,Ez,Bx,By,Bz) |
|---|---|---|---|---|---|---|
| 2 | 2x | Ey | 5.293e-03 | 4.567e-09 | 8.628e-07 | 5.2928e-03, 5.2928e-03, 1.1097e-03, 3.2850e-03, 3.2850e-03, 0.0000e+00 |
| 2 | 3x | Ex | 5.293e-03 | 4.567e-09 | 8.628e-07 | 5.2928e-03, 5.2928e-03, 1.1097e-03, 3.2850e-03, 3.2850e-03, 0.0000e+00 |
| 4 | 2x | Ex | 1.028e-03 | 2.484e-10 | 2.416e-07 | 1.0284e-03, 1.0284e-03, 2.4980e-04, 7.4500e-04, 7.4500e-04, 0.0000e+00 |
| 4 | 3x | Ex | 1.028e-03 | 2.484e-10 | 2.416e-07 | 1.0284e-03, 1.0284e-03, 2.4980e-04, 7.4500e-04, 7.4500e-04, 0.0000e+00 |
| 8 | 2x | Ey | 1.320e-04 | 6.699e-12 | 5.076e-08 | 1.3197e-04, 1.3197e-04, 3.6258e-05, 1.0854e-04, 1.0854e-04, 0.0000e+00 |
| 8 | 3x | Ey | 1.320e-04 | 6.702e-12 | 5.076e-08 | 1.3201e-04, 1.3202e-04, 3.6258e-05, 1.0854e-04, 1.0854e-04, 0.0000e+00 |
| 16 | 2x | Ex | 1.236e-05 | 1.065e-13 | 8.613e-09 | 1.2365e-05, 1.2362e-05, 3.7438e-06, 1.1224e-05, 1.1224e-05, 0.0000e+00 |
| 16 | 3x | Ex | 1.240e-05 | 1.068e-13 | 8.613e-09 | 1.2404e-05, 1.2339e-05, 3.7438e-06, 1.1224e-05, 1.1224e-05, 0.0000e+00 |
| 32 | 2x | Bx | 2.588e+02 | 3.155e-07 | -1.219e-09 | 1.8885e+02, 1.8885e+02, 4.1353e+01, 2.5880e+02, 2.5880e+02, 0.0000e+00 |
| 32 | 3x | Ez | 2.831e+00 | 2.380e-07 | 8.407e-08 | 1.2383e+00, 1.2383e+00, 2.8314e+00, 1.4791e+00, 1.4791e+00, 0.0000e+00 |
| 48 | 2x | Ey | 1.258e+02 | 5.037e-08 | 4.002e-10 | 1.2585e+02, 1.2585e+02, 2.2284e+01, 1.1173e+02, 1.1173e+02, 0.0000e+00 |
| 48 | 3x | Bx | 1.517e+02 | 5.892e-08 | -3.884e-10 | 1.2585e+02, 1.2585e+02, 2.9488e+01, 1.5169e+02, 1.5169e+02, 0.0000e+00 |
### single_g100_perp
| d (cells) | pad | comp worst | rel err | abs err | ref value | per-component rel (Ex,Ey,Ez,Bx,By,Bz) |
|---|---|---|---|---|---|---|
| 2 | 2x | Ey | 1.763e+01 | 5.581e-01 | 3.166e-02 | 1.4125e+01, 1.7630e+01, 4.4481e-01, 9.8863e-01, 9.8587e-01, 0.0000e+00 |
| 2 | 3x | Ey | 1.763e+01 | 5.581e-01 | 3.166e-02 | 1.4125e+01, 1.7630e+01, 4.4481e-01, 9.8863e-01, 9.8587e-01, 0.0000e+00 |
| 4 | 2x | Ey | 8.537e+00 | 1.556e-01 | 1.823e-02 | 7.5839e+00, 8.5375e+00, 3.4236e-01, 9.7490e-01, 9.7082e-01, 0.0000e+00 |
| 4 | 3x | Ey | 8.537e+00 | 1.556e-01 | 1.823e-02 | 7.5839e+00, 8.5375e+00, 3.4236e-01, 9.7490e-01, 9.7082e-01, 0.0000e+00 |
| 8 | 2x | Ey | 4.225e+00 | 3.034e-02 | 7.180e-03 | 3.9794e+00, 4.2253e+00, 1.6744e-01, 9.3256e-01, 9.2618e-01, 0.0000e+00 |
| 8 | 3x | Ey | 4.225e+00 | 3.034e-02 | 7.180e-03 | 3.9794e+00, 4.2253e+00, 1.6744e-01, 9.3256e-01, 9.2618e-01, 0.0000e+00 |
| 16 | 2x | Ey | 2.005e+00 | 4.298e-03 | 2.144e-03 | 1.9337e+00, 2.0045e+00, 1.0033e-01, 8.0278e-01, 7.9346e-01, 0.0000e+00 |
| 16 | 3x | Ey | 2.005e+00 | 4.298e-03 | 2.144e-03 | 1.9337e+00, 2.0045e+00, 1.0033e-01, 8.0278e-01, 7.9346e-01, 0.0000e+00 |
| 32 | 2x | Ey | 3.220e+02 | 1.733e-01 | 5.382e-04 | 7.0896e+01, 3.2201e+02, 1.3205e+00, 1.2281e-01, 1.1448e+00, 0.0000e+00 |
| 32 | 3x | Ex | 4.227e+00 | 1.519e-01 | 3.594e-02 | 4.2268e+00, 2.9853e+00, 6.6858e-01, 3.7279e-01, 1.4841e+00, 0.0000e+00 |
| 48 | 2x | Ey | 1.688e+02 | 3.730e-02 | 2.209e-04 | 3.0335e+01, 1.6884e+02, 2.5752e+00, 9.6369e-01, 5.8495e-01, 0.0000e+00 |
| 48 | 3x | Ey | 1.688e+02 | 3.730e-02 | 2.209e-04 | 3.9306e+01, 1.6884e+02, 2.5752e+00, 9.6369e-01, 1.3799e+00, 0.0000e+00 |
### cloud100_g100_axial
| d (cells) | pad | comp worst | rel err | abs err | ref value | per-component rel (Ex,Ey,Ez,Bx,By,Bz) |
|---|---|---|---|---|---|---|
| 2 | 2x | Ey | 1.433e-03 | 7.149e-10 | 4.990e-07 | 2.5744e-04, 1.4328e-03, 4.6960e-05, 7.8131e-04, 1.3375e-05, 0.0000e+00 |
| 2 | 3x | Ey | 1.433e-03 | 7.149e-10 | 4.990e-07 | 2.5744e-04, 1.4328e-03, 4.6960e-05, 7.8131e-04, 1.3375e-05, 0.0000e+00 |
| 4 | 2x | Ey | 2.256e-04 | 3.852e-11 | 1.708e-07 | 1.1190e-04, 2.2555e-04, 9.0513e-06, 1.6857e-04, 1.0789e-04, 0.0000e+00 |
| 4 | 3x | Ey | 2.255e-04 | 3.852e-11 | 1.708e-07 | 1.1190e-04, 2.2555e-04, 9.0513e-06, 1.6857e-04, 1.0789e-04, 0.0000e+00 |
| 8 | 2x | Ex | 5.278e-05 | 2.257e-12 | 4.276e-08 | 5.2784e-05, 4.5888e-05, 1.4319e-06, 4.0281e-05, 4.7440e-05, 0.0000e+00 |
| 8 | 3x | Ex | 5.279e-05 | 2.258e-12 | 4.276e-08 | 5.2791e-05, 4.5888e-05, 1.4319e-06, 4.0281e-05, 4.7440e-05, 0.0000e+00 |
| 16 | 2x | Ez | 1.573e+00 | 3.428e-07 | 2.179e-07 | 7.5227e-01, 1.9283e-01, 1.5730e+00, 2.8402e-01, 9.9094e-01, 0.0000e+00 |
| 16 | 3x | Ex | 1.273e-05 | 1.243e-13 | 9.762e-09 | 1.2730e-05, 9.9984e-06, 2.0185e-07, 9.3228e-06, 1.1817e-05, 0.0000e+00 |
| 32 | 2x | Ex | 1.298e+09 | 2.319e+00 | 1.786e-09 | 1.2984e+09, 5.9465e+08, 5.1773e+02, 1.1511e+04, 1.1189e+05, 0.0000e+00 |
| 32 | 3x | By | 4.485e+00 | 7.697e-09 | 1.716e-09 | 3.9128e+00, 2.4555e+00, 3.8773e+00, 2.8424e+00, 4.4852e+00, 0.0000e+00 |
| 48 | 2x | Ey | 8.746e+01 | 4.409e-08 | 5.042e-10 | 6.8930e+01, 8.7457e+01, 1.5252e+01, 7.8955e+01, 6.2832e+01, 0.0000e+00 |
| 48 | 3x | Bx | 2.882e+03 | 1.412e-06 | -4.898e-10 | 1.3565e+03, 7.5102e+02, 1.2365e+02, 2.8821e+03, 2.5019e+03, 0.0000e+00 |
### cloud100_g100_perp
| d (cells) | pad | comp worst | rel err | abs err | ref value | per-component rel (Ex,Ey,Ez,Bx,By,Bz) |
|---|---|---|---|---|---|---|
| 2 | 2x | Ey | 5.868e+00 | 3.751e-02 | 6.392e-03 | 4.9324e+00, 5.8679e+00, 1.7032e-01, 9.4749e-01, 9.4050e-01, 0.0000e+00 |
| 2 | 3x | Ey | 5.868e+00 | 3.751e-02 | 6.392e-03 | 4.9324e+00, 5.8679e+00, 1.7032e-01, 9.4749e-01, 9.4050e-01, 0.0000e+00 |
| 4 | 2x | Ey | 4.409e+00 | 1.820e-02 | 4.128e-03 | 3.7962e+00, 4.4090e+00, 8.4765e-02, 9.2232e-01, 9.1287e-01, 0.0000e+00 |
| 4 | 3x | Ey | 4.409e+00 | 1.820e-02 | 4.128e-03 | 3.7962e+00, 4.4090e+00, 8.4765e-02, 9.2232e-01, 9.1287e-01, 0.0000e+00 |
| 8 | 2x | Ey | 2.829e+00 | 5.893e-03 | 2.083e-03 | 2.5007e+00, 2.8289e+00, 2.7008e-02, 8.5996e-01, 8.4593e-01, 0.0000e+00 |
| 8 | 3x | Ey | 2.829e+00 | 5.893e-03 | 2.083e-03 | 2.5007e+00, 2.8289e+00, 2.7008e-02, 8.5996e-01, 8.4593e-01, 0.0000e+00 |
| 16 | 2x | Ex | 4.476e+00 | 9.229e-02 | 2.062e-02 | 4.4765e+00, 2.8011e+00, 3.2311e-01, 6.8165e-01, 1.1681e+00, 0.0000e+00 |
| 16 | 3x | Ey | 1.468e+00 | 1.197e-03 | 8.157e-04 | 1.3290e+00, 1.4678e+00, 1.9867e-01, 7.0401e-01, 6.8464e-01, 0.0000e+00 |
| 32 | 2x | Ey | 2.029e+03 | 5.206e-01 | 2.566e-04 | 6.5376e+02, 2.0286e+03, 1.3881e+00, 1.5376e-01, 1.0361e+00, 0.0000e+00 |
| 32 | 3x | Ex | 5.887e+00 | 6.848e-02 | 1.163e-02 | 5.8867e+00, 5.6562e+00, 9.2158e-01, 1.0853e-01, 1.5284e+00, 0.0000e+00 |
| 48 | 2x | Ey | 6.566e+01 | 7.856e-03 | 1.197e-04 | 1.3322e+01, 6.5655e+01, 2.0173e+00, 1.1776e+00, 3.9188e-01, 0.0000e+00 |
| 48 | 3x | Ey | 2.665e+02 | 3.189e-02 | 1.197e-04 | 2.0065e+02, 2.6650e+02, 2.3262e+00, 1.3634e+00, 1.2342e+00, 0.0000e+00 |

### Where the error comes from
| case | first d with rel>1e-3 (2x / 3x) | first d where 2x is >3x worse than 3x (padding-driven) | worst rel over sweep (2x / 3x) |
|---|---|---|---|
| single_g100_axial | 2 / 2 | 32 | 2.588e+02 / 1.517e+02 |
| single_g100_perp | 2 / 2 | 32 | 3.220e+02 / 1.688e+02 |
| cloud100_g100_axial | 2 / 2 | 16 | 1.298e+09 / 2.882e+03 |
| cloud100_g100_perp | 2 / 2 | 16 | 2.029e+03 / 2.665e+02 |

Reading: the per-component worst ratio reaches 1e-3 at d=2 in every case because a small transverse component (Ey on the beam axis) is scaled by its own tiny magnitude.  The padding-driven error is separated in the third column: for the clouds 2x starts to lose to 3x at d=16, for the single particle at d=32.  That is the point where the read-out node moves beyond M/2 from the source nodes, i.e. where the cyclic convolution substitutes the nearest periodic image.  In the perpendicular direction the error at d<=16 is identical for 2x and 3x to the printed digits: it is quadrature-driven (TSC deposit + 1-cell kernel sampling cannot resolve the d/gamma angular scale), see section 4.
Padding design rule implied by the sweep: to read out to d_max cells beyond a source box of n0 cells, M/2 must exceed n0 + d_max, i.e. pad > 2*(1 + d_max/n0).  For n0=20 and d_max=48 that is pad > 6.8, so 8x, not 2x or 3x.

## 4. Kernel sampling step vs cell size
One particle at (0.3,0.2,0.1); 10 observation points on a shell of radius 8 cells (|R| >= 6.5 to every source node, i.e. the mesh/shell boundary region).  Mesh spacing delta = h/m; the TSC source shape keeps its physical width of one cell; the kernel is sampled exactly at the mesh nodes (step delta).  The sum is direct, so only the kernel sampling enters.  Reference: indfield.py order 20 refine 0.
### stationary
Reference self-check (point 6, order20/ref0 vs order24/ref1): 5.480e-15 relative
| m = h/delta | delta/h | nonzero source nodes | comp worst | rel err | abs err | maxabs/max|ref| |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 27 | Ex | 4.424e-04 | 7.451e-06 | 4.424e-04 |
| 2 | 0.500 | 216 | Ez | 4.443e-05 | 7.096e-07 | 4.213e-05 |
| 4 | 0.250 | 1728 | Ex | 6.286e-06 | 1.059e-07 | 6.286e-06 |
| 8 | 0.125 | 13824 | Ez | 7.505e-07 | 1.199e-08 | 7.117e-07 |
| 16 | 0.062 | 110592 | Ex | 9.822e-08 | 1.654e-09 | 9.822e-08 |

Convergence ratios e(1)/e(2)=10.0, e(2)/e(4)=7.1, e(4)/e(8)=8.4, e(8)/e(16)=7.6
### gamma100 (u along z)
Reference self-check (point 6, order20/ref0 vs order24/ref1): 2.609e-15 relative
| m = h/delta | delta/h | nonzero source nodes | comp worst | rel err | abs err | maxabs/max|ref| |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 27 | By | 5.645e+00 | 1.067e+00 | 5.645e+00 |
| 2 | 0.500 | 216 | By | 2.337e+00 | 4.415e-01 | 2.337e+00 |
| 4 | 0.250 | 1728 | By | 7.469e-01 | 1.411e-01 | 7.469e-01 |
| 8 | 0.125 | 13824 | Ez | 2.693e-01 | 2.236e-02 | 1.184e-01 |
| 16 | 0.062 | 110592 | Ez | 3.244e-01 | 2.127e-05 | 1.126e-04 |

Convergence ratios e(1)/e(2)=2.4, e(2)/e(4)=3.1, e(4)/e(8)=2.8, e(8)/e(16)=0.8
Rule, stationary field: delta = h already meets the 1e-3 gate (4.4e-04); the error falls by ~8x per halving of delta (third order, the TSC shape is piecewise quadratic).  delta = h/2 gives 4.4e-05.
Rule, gamma=100 beamed field: delta = h is far outside the gate (per-component 5.6e+00) and halving is not enough.  The limiter is the angular scale d/gamma, not the TSC shape: at distance d the kernel has to resolve a transverse structure of ~d/gamma cells, so the required step is delta <= d/(gamma*j), j ~ 1.  At d=8, gamma=100 that is delta <= 0.08 = h/12.5.  The last two rows show the per-component ratio stalling near 0.3 for Ez while the absolute error keeps falling (Ez crosses zero there, so the component-wise ratio is not meaningful at that level; the maxabs/max|ref| column is the honest measure).  For an h=1 mesh the beamed field is unreachable at any usable distance; the near shell must extend over ~gamma*h cells, not a few.

## 5. Cost model, CPU only (20-cell bunch box)
Wall time per stage, median of 5, CPU pocketfft with OMP_NUM_THREADS=16.  The kernel build (point kernel sampled on the padded grid, 6 components) is reported separately from the transforms so the two are not conflated.
| pad | M^3 | complex128 array | deposit ms | kernel build ms | C2C fwd+prod+inv ms | R2C fwd+prod+inv ms | R2C/C2C |
|---|---|---|---|---|---|---|---|
| 2x | 64000 | 1.0 MB | 8.73 | 33.2 | 2.6 | 1.9 | 0.74 |
| 3x | 216000 | 3.5 MB | 8.83 | 115.9 | 6.7 | 4.2 | 0.62 |

Cost model: 13 transforms of M^3 complex128 samples (1 forward density + 6 forward kernels + 6 inverse), about 5*M^3*log2(M) flops each, and 16*M^3 bytes per complex128 working array (plus one more for the 6 kernel spectra, which can be streamed).  Scaling from 2x to 3x is (60/40)^3 log(60)/log(40) = 3.75 for the transform; measured 2.59. R2C/C2R halves the last axis: ~half the memory; on this CPU pocketfft the speed-up is only 0.74x because the transform is memory-bound at these small sizes.  These are CPU numbers, not GPU numbers.

## 6. Summary, recommended defaults, open items

Host/versions: rosi_v100, CPU only (no GPU used), OMP_NUM_THREADS=16, Python 3.12.3,
numpy 2.5.4, scipy 1.18.1. Independent truth: ~/pr24/pr24w53-ref/indfield.py.
No prior N1 script existed on rosi; this suite was built from scratch.

### 6a. The five answers

1. Wraparound (worst over the six components, 20^3 box, vs the direct non-cyclic sum):
   1x = 9.275e-3 (single stationary, Ey), 1.34e-1 (100-source cloud, Ez);
   1.5x = 1.9e-16 for a single particle away from the wall, but 5.7e-4 / 6.9e-3 when the
   source touches the box (clouds);
   2x = 4.8e-16, 3x = 3.2e-16, 4x = 5.6e-16 (machine precision).
   The 1x single-particle value reproduces the documented 8.947e-3 to 3.7%.

2. R2C/C2R vs C2C: max abs difference 1.665e-16 over 8000 nodes and 6 components;
   per-component worst relative difference 1.5e-16 .. 4.5e-16. The Hermitian-mirrored
   half-spectrum product matches the full-spectrum product to 3.6e-15; the self-conjugate
   bins (DC and the 7 even/even/even Nyquist corners) have max |Im| 2.6e-17, while the
   kx-Nyquist plane is Hermitian-paired, not real (max |Im| 0.13 there). See section 2
   for the exact bin rule.

3. Padding vs distance: see the four tables in section 3. Padding-driven divergence begins
   at d=16 for the 100-source clouds and at d=32 for the single particle (first d where 2x
   is more than 3x worse than 3x). For d<=8 the 2x and 3x results agree to the printed
   digits in every case, so the residual there is quadrature-driven, not padding-driven.
   Design rule: M/2 > n0 + d_max, i.e. pad > 2*(1 + d_max/n0); for n0=20 and d_max=48 that
   is pad > 6.8 (8x), not 2x or 3x.

4. Kernel sampling: stationary field at |R|>=6.5 cells, delta=h gives 4.4e-4 (meets the
   1e-3 gate), delta=h/2 gives 4.4e-5; ratios 10.0/7.1/8.4/7.6 per halving (about third
   order, the TSC shape is piecewise quadratic). gamma=100 beamed field at r=8:
   delta=h 5.6, h/2 2.34, h/4 0.75, h/8 0.118, h/16 1.1e-4 (scale-relative). The limiter is
   the angular scale d/gamma: delta <= d/(gamma*1.25) at d=8, gamma=100.

5. Cost (CPU, 20-cell bunch box): 2x -> M=40^3 (64000 nodes, 1.0 MB per complex128 array):
   deposit 8.7-9.0 ms, kernel build 32.8-33.2 ms, C2C 2.5-2.6 ms, R2C 1.9 ms. 3x ->
   M=60^3 (216000 nodes, 3.5 MB): deposit 8.8-9.7 ms, kernel build 112-116 ms, C2C 6.7-6.8 ms,
   R2C 4.2 ms. The 2x->3x transform scaling is 2.6-3.1 (model 3.75). R2C is 1.35x (2x) /
   1.6x (3x) faster than C2C on this CPU. Run-to-run spread on this shared node is ~10%.
   All numbers are CPU numbers, not GPU numbers.

### 6b. Recommended picBunch defaults

- fftPadding: 2 when the written field covers only the bunch box (machine precision
  measured). Otherwise compute it from the write-out extent: pad > 2*(1 + d_max/n0), and
  never below 2. Never 1x (9.3e-3 measured) and do not assume 1.5x is safe (up to 6.9e-3).
- R2C/C2R for all six components (identical result, half the memory). Keep the spectra of
  the six kernels; they depend only on the velocity and the padded grid, so build them once
  per (velocity, geometry) and reuse them across events. On CPU the kernel build costs
  13x/16x the transform, so caching matters more than the FFT itself.
- Double precision for the deposit, kernel and FFT (C2C/R2C agreement 1.7e-16 leaves the
  whole budget to the quadrature).
- Do not use the mesh alone for gamma=100. The matched point kernel fails by 14x at d=2
  perpendicular; the TSC-convolved kernel (the documented choice) cuts that to 0.39 at
  d=2, 0.23 at d=8, 0.20 at d=16 - better, still far above 1e-3. The near shell must cover
  ~gamma*h cells at h=1, or the mesh must be refined by ~gamma/10, or the sampled kernel
  must resolve d/gamma (section 4).
- Use the TSC-convolved kernel, not the bare point kernel, whenever the mesh is used for a
  beamed source: it is the difference between a 14x and a 0.4x error in the near band.

### 6c. Control not in the main tables

Single gamma=100 particle, perpendicular read-out, TSC-convolved kernel vs the matched
point kernel (scale-relative max error over the six components):
d=2: point kernel 1.41e+01, TSC-convolved 3.90e-01; d=8: 3.98e+00 vs 2.26e-01;
d=16: 1.93e+00 vs 1.97e-01. The main section-3 tables use the matched point kernel because
it is the consistent kernel for a TSC-deposited source (the TSC-convolved kernel convolves
the shape twice). Both kernels miss the 1e-3 gate in the beamed perpendicular direction.

### 6d. Open items

- The full TSC-convolved-kernel distance sweep with the 100-source cloud was not run
  (100 particles x 27 source nodes x 46656 quadrature nodes per component is too costly
  here); only the single-particle control above was measured.
- The documented "1e-8 in the far field" claim is not reproduced for a gamma=100 beam at
  d<=48 cells and h=1 with either kernel; my reading is that it applies to non-beamed
  fields or to d >> gamma*h. This needs an explicit reconciliation with the author.
- exp3 uses the periodic fold p mod M with the kernel sampled on the M grid (what a real
  zero-padded FFT does), not a periodized kernel; the folding convention is stated in
  section 3.
- No GPU timings were taken, by instruction.

### 6e. Exact commands

    # on rosi_v100, CPU only
    cd /home/ubuntu/pr24/n1-20261102
    export OMP_NUM_THREADS=16
    python3 n1_measure.py init       # version header, truncates RESULTS.md
    python3 n1_measure.py 1          # wraparound control
    python3 n1_measure.py 2          # C2C vs R2C, bin rule
    python3 n1_measure.py 3 <c> <d>  # one exp3 cell, c in 0..3, d in {2,4,8,16,32,48}
    python3 n1_measure.py report3    # build the section-3 tables from exp3.csv
    python3 n1_measure.py 4          # kernel sampling rule
    python3 n1_measure.py 5          # CPU cost model
