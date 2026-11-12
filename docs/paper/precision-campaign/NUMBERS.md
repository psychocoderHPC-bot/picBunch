# NUMBERS - fresh precision values

Fresh campaign data: jureca-h100, 4x H100 PCIe 80 GB, build `00fb8f7`, harness `f8d7f2b`, binary `717212e`.
`fresh=null` means no fresh backing (external/archived data, structural value, or var1b).
`committed` is the pre-T7a data value where a data file exists; `rel` is `(fresh-paper)/paper`.

| tex:line | id | paper | fresh | committed | rel | unit | figure |
|---|---|---|---|---|---|---|---|
| app_multivel.tex:38 | multivel_energy_Bz_b16 | 2.32e+04 | 2.328e+04 | 2.32e+04 | +0.31% | tesla | F_multivel_dist |
| app_multivel.tex:38 | multivel_energy_Bz_b64 | 2.32e+04 | 2.328e+04 | 2.32e+04 | +0.31% | tesla | F_multivel_dist |
| app_multivel.tex:39 | multivel_angular_By_b16 | 1295 | 1285 | 1297 | -0.75% | tesla | F_multivel_dist |
| app_multivel.tex:39 | multivel_angular_By_b64 | 1296 | 1286 | 1297 | -0.70% | tesla | F_multivel_dist |
| app_multivel.tex:39 | multivel_angular_Bz_b16 | 2.482e+04 | 2.489e+04 | 2.466e+04 | +0.27% | tesla | F_multivel_dist |
| app_multivel.tex:39 | multivel_angular_Bz_b64 | 2.482e+04 | 2.489e+04 | 2.466e+04 | +0.27% | tesla | F_multivel_dist |
| app_multivel.tex:55-56 | multivel_dev_angular_Bz | 4.4e-05 | 2.331e-05 | 5.176e-05 | -47.03% | relative | F_multivel_dist |
| app_multivel.tex:55 | multivel_dev_energy_Bz | 4.3e-06 | 6.89e-06 | 2.698e-06 | +60.23% | relative | F_multivel_dist |
| app_multivel.tex:56 | multivel_dev_angular_By | 0.00056 | 0.001066 | 0.0002886 | +90.30% | relative | F_multivel_dist |
| app_multivel.tex:59 | multivel_ratio_By_Bz | 0.0522 | 0.05163 | 0.05258 | -1.08% | ratio | F_multivel_dist |
| app_multivel.tex:71 | multivel_agree_By_pct | 0.056 | 0.1066 | 0.02886 | +90.30% | percent | F_multivel_dist |
| app_multivel.tex:71 | multivel_agree_Bz_pct | 0.0043 | 0.002331 | 0.005176 | -45.79% | percent | F_multivel_dist |
| app_multivel.tex:88-98 | var1b_peaks | [7.83, 2.33e+04, 7.76, 2.34e+04, 7.56, 7.52, 2.32e+04, 2.33e+04, 8.73] | - | - | - | mixed | - |
| app_multivel.tex:106-142 | far_tail_margin | - | - | - | - | mixed | - |
| app_slices.tex:37-38 | insupport_16pct | 1.6 | - | - | - | percent | - |
| app_slices.tex:156-175 | tab_slices | - | - | - | - | mixed | - |
| app_slices.tex:198-219 | nearshell_ranges | - | - | - | - | mixed | - |
| app_slices.tex:271 | large_ex_xy_pct | 0.5475 | 0.5475 | 0.5475 | +0.01% | percent of shared E maximum | large_sxy_g10 |
| app_slices.tex:271 | large_ex_zy_pct | 0.1385 | 0.1384 | 0.1384 | -0.04% | percent of shared E maximum | large_szy_g10 |
| app_slices.tex:277 | large_b_xy | 0.00015 | 0.0001468 | 0.0001468 | -2.13% | tesla | large_sxy_g10 |
| app_slices.tex:277 | large_b_zy | 0.00014 | 0.0001406 | 0.0001406 | +0.43% | tesla | large_szy_g10 |
| app_slices.tex:305 | large_g1_core | 0.84 | 0.8441 | 0.8441 | +0.49% | relative error | large_szy_g1 |
| app_sweep.tex:20-24 | floor_band | 0.0035 | - | - | - | relative error | - |
| app_sweep.tex:38 | hires_worst_g1p0 | 0.003474 | 0.003779 (full 0.0701) | 0.0005769 | +8.78% | relative error | gp_g1p0 |
| app_sweep.tex:39 | hires_worst_g1p1 | 0.003669 | 0.003747 (full 0.06745) | 0.0006098 | +2.12% | relative error | gp_g1p1 |
| app_sweep.tex:40 | hires_worst_g1p5 | 0.003719 | 0.002578 (full 0.07726) | 0.0006305 | -30.68% | relative error | gp_g1p5 |
| app_sweep.tex:41 | hires_worst_g2p0 | 0.005098 | 0.007078 (full 0.09569) | 0.001343 | +38.84% | relative error | gp_g2p0 |
| app_sweep.tex:42 | hires_worst_g10 | 0.1197 | 0.3505 (full 0.674) | 0.1197 | +192.82% | relative error | gp_g10 |
| app_sweep.tex:43 | hires_worst_g50 | 1.426 | 1.59 (full 7.095) | 1.426 | +11.47% | relative error | gp_g50 |
| app_sweep.tex:44 | hires_worst_g100 | 3.213 | 3.49 (full 4.263) | 3.213 | +8.62% | relative error | gp_g100 |
| main.tex:48-49 | crossover_d48_g10 | 0.00026 | - | - | - | relative error | - |
| main.tex:48-49 | gate_3e-4 | 0.0003 | - | - | - | relative error | - |
| sec_accuracy.tex:30-34 | b_over_e_peak_t | 2.3e+04 | 2.328e+04 | 2.32e+04 | +1.21% | tesla | F_multivel_dist |
| sec_accuracy.tex:30-34 | b_over_e_peak_vm | 7e+12 | 7.037e+12 | 7.022e+12 | +0.53% | V/m | F_multivel_dist |
| sec_accuracy.tex:30-34 | b_over_e_ratio | 3.3e-09 | - | - | - | ratio | F_multivel_dist |
| sec_accuracy.tex:70 | d48_mt_g1 | 1.38e-07 | - | - | - | relative error | - |
| sec_accuracy.tex:70 | d48_mt_g10 | 0.00026 | - | - | - | relative error | - |
| sec_accuracy.tex:71 | multivel_1mrad_worst | 1.4 | - | - | - | percent | val_1mrad_xy_E |
| sec_accuracy.tex:71 | multivel_binning_rel_pct | 0.00043 | 0.000689 | 0.0002698 | +60.23% | percent | F_multivel_dist |
| sec_accuracy.tex:72 | fast_axial | 0.00014 | - | - | - | relative error | - |
| sec_accuracy.tex:72 | fast_oblique | 0.00194 | - | - | - | relative error | - |
| sec_accuracy.tex:72 | fast_stationary | 0.000371 | - | - | - | relative error | - |
| sec_accuracy.tex:73 | far_tail_512 | 0.0028 | - | - | - | V/m | - |
| sec_accuracy.tex:73 | far_tail_6r | 0.0853 | - | - | - | V/m | - |
| sec_accuracy.tex:73 | far_tail_frac | 0.00187 | - | - | - | fraction of tolerance | - |
| sec_accuracy.tex:94-96 | shape_bias_cic | [-0.00831, -0.000562, -3.55e-05] | - | - | - | relative error | - |
| sec_accuracy.tex:94-96 | shape_bias_pqs | [-0.00831, -0.000562, -3.55e-05] | - | - | - | relative error | - |
| sec_accuracy.tex:94-96 | shape_bias_tsc | [-0.00831, -0.000562, -3.55e-05] | - | - | - | relative error | - |
| sec_accuracy.tex:108-111 | d48_l2_ratio | 58 | - | - | - | ratio | - |
| sec_accuracy.tex:108-111 | d48_single_g10a1 | 0.0014 | - | - | - | relative error | - |
| sec_accuracy.tex:108-111 | d48_single_g1a4 | 2.65e-05 | - | - | - | relative error | - |
| sec_accuracy.tex:121-126 | tab_d48 | - | - | - | - | relative error | - |
| sec_accuracy.tex:156 | w4b_caption_d2 | 0.12 | 0.1165 | 0.1165 | -2.91% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:156 | w4b_caption_d32 | 1.4e-06 | 1.618e-07 | 1.429e-06 | -88.44% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:167 | w4b_text_d12_g1 | 0.00011 | 0.000109 | 0.000109 | -0.93% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:168 | w4b_text_d32_g10 | 0.0063 | 0.006268 | 0.006268 | -0.51% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:169 | w4b_text_d32_g100 | 0.47 | 0.4642 | 0.4688 | -1.24% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:193 | w4b_tab_1_1_16 | 3.3e-05 | 8.634e-06 | 3.322e-05 | -73.83% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:193 | w4b_tab_1_1_32 | 1.4e-06 | 1.618e-07 | 1.429e-06 | -88.44% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:194 | w4b_tab_1_4_16 | 3.3e-05 | 8.634e-06 | 3.322e-05 | -73.83% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:194 | w4b_tab_1_4_32 | 1.4e-06 | 1.618e-07 | 1.429e-06 | -88.44% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:195 | w4b_tab_10_1_16 | 0.058 | 0.05593 | 0.05767 | -3.58% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:195 | w4b_tab_10_1_32 | 0.0063 | 0.006268 | 0.006268 | -0.51% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:196 | w4b_tab_10_4_16 | 0.058 | 0.05593 | 0.05767 | -3.58% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:196 | w4b_tab_10_4_32 | 0.0063 | 0.006268 | 0.006268 | -0.51% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:197 | w4b_tab_50_1_16 | 0.47 | 0.469 | 0.469 | -0.21% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:197 | w4b_tab_50_1_32 | 0.25 | 0.2466 | 0.2466 | -1.37% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:198 | w4b_tab_50_4_16 | 0.47 | 0.469 | 0.469 | -0.21% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:198 | w4b_tab_50_4_32 | 0.25 | 0.2466 | 0.2466 | -1.37% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:199 | w4b_tab_100_1_16 | 0.7 | 0.702 | 0.7029 | +0.28% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:199 | w4b_tab_100_1_32 | 0.47 | 0.4642 | 0.4688 | -1.24% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:200 | w4b_tab_100_4_16 | 0.7 | 0.702 | 0.7029 | +0.28% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:200 | w4b_tab_100_4_32 | 0.47 | 0.4642 | 0.4688 | -1.24% | relative error | F_w4b_rel_error |
| sec_accuracy.tex:226-237 | tab_crossover | - | - | - | - | cells | - |
| sec_accuracy.tex:245-250 | crossover_axial_d6 | 6 | - | - | - | cells | - |
| sec_accuracy.tex:245-250 | crossover_d48_narrative | 0.00026 | - | - | - | relative error | - |
| sec_accuracy.tex:267 | gamma_slope_p10 | 1.64 | 1.664 | 1.639 | +1.49% | exponent | F_gamma_err_dist |
| sec_accuracy.tex:274 | gamma_tab_10_0p25_d8 | 0.00025 | 0.0002565 | 0.0002539 | +2.61% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:274 | gamma_tab_10_1p0_d8 | 0.044 | 0.04673 | 0.04443 | +6.20% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:275 | gamma_slope_p50 | 1.71 | 1.307 | 1.705 | -23.59% | exponent | F_gamma_err_dist |
| sec_accuracy.tex:275 | gamma_tab_50_0p25_d8 | 0.074 | 0.07277 | 0.07359 | -1.66% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:275 | gamma_tab_50_1p0_d8 | 1.05 | 0.388 | 1.052 | -63.04% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:276 | gamma_slope_p100 | 2.06 | 1.23 | 2.064 | -40.27% | exponent | F_gamma_err_dist |
| sec_accuracy.tex:276 | gamma_tab_100_0p25_d8 | 0.21 | 0.1838 | 0.2108 | -12.48% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:276 | gamma_tab_100_1p0_d8 | 5.47 | 0.9322 | 5.472 | -82.96% | peak-normalised absolute error | F_gamma_err_dist |
| sec_accuracy.tex:361-368 | fast_closest_node | [0.0001, 0.001, 0.03, 0.016] | - | - | - | mixed | - |
| sec_consistency.tex:11-15 | coulomb_elem | 2.28e-25 | - | - | - | absolute residual | - |
| sec_consistency.tex:11-15 | coulomb_l2 | 2.29e-25 | - | - | - | absolute residual | - |
| sec_consistency.tex:11-15 | coulomb_max | 4.88e-14 | - | - | - | relative error | - |
| sec_consistency.tex:22-25 | consistency_contraction_g10 | 0.12 | 0.3505 (full 0.674) | 0.1197 | +192.09% | relative error | gp_g10 |
| sec_consistency.tex:22-25 | consistency_contraction_g100 | 3.21 | 3.49 (full 4.263) | 3.213 | +8.72% | relative error | gp_g100 |
| sec_consistency.tex:22-25 | consistency_contraction_g50 | 1.43 | 1.59 (full 7.095) | 1.426 | +11.16% | relative error | gp_g50 |
| sec_consistency.tex:32-38 | gauss_indicator | [0.353, 5.03, 0.0904, 0.0814, 0.0895, 0.399, 12.6, 4.9] | - | - | - | mixed | - |
| sec_method.tex:209-214 | padding | - | - | - | - | mixed | - |
| sec_validation.tex:16-20 | crossval_318 | [0.0486, 0.62, 0.516] | - | - | - | relative error | - |
| sec_validation.tex:52-61 | wholefield | - | - | - | - | mixed | - |
