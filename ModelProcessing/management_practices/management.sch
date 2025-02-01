management.sch file AMES
                          NAME NUMB_OPS NUMB_AUTO OP_TYP        MON        DAY HU_SCH   OP_DATA1    OP_DATA2  OP_DATA3
mgt_01                              37         1
                                                irr_opt_sw_unlim
                                                    till          4          1    0.0    chisplow        null     0.000
                                                    plnt          4         15    0.0        corn        null         0
                                                    fert          5          1    0.0      elem_n   broadcast   120.000
                                                    hvkl          9          1    0.0        corn       grain     0.000
                                                    fert          9          9    0.0    11_52_00   broadcast   110.000
                                                    till          9         15    0.0    chisplow        null     0.000													
                                                    skip          0          0      0        null        null         0													

                                                    till          5          1    0.0     fldcult        null     0.000
                                                    plnt          5         3    0.0        corn        null         0
                                                    fert          5         15    0.0      elem_n   broadcast    80.000
                                                    fert          6          1    0.0      elem_n   broadcast   150.000												

                                                    hvkl          9          5    0.0        corn       grain     0.000
                                                    fert          9         14    0.0    05_10_15   broadcast   240.000												

                                                    skip          0          0      0        null        null         0
                                                    till          4         20    0.0     fldcult        null     0.000
                                                    fert          4         25    0.0      elem_n   broadcast    80.000
                                                    plnt          4         25    0.0        corn        null         0
                                                    fert          6         10    0.0      elem_n   broadcast   150.000												

                                                    hvkl          9          7    0.0        corn       grain     0.000
                                                    fert          9         12    0.0    11_15_00   broadcast   140.000	

                                                    till          9         16    0.0    chisplow        null     0.000												

                                                    skip          0          0      0        null        null         0											

                                                    till          4         16    0.0     fldcult        null     0.000
                                                    till          4         30    0.0     fldcult        null     0.000
                                                    plnt          5          6    0.0        corn        null         0
                                                    fert          6         17    0.0      elem_n   broadcast   110.000												

                                                    hvkl          9          4    0.0        corn       grain     0.000
                                                    fert         12         21    0.0    11_52_00   broadcast   130.000											

                                                    skip          0          0      0        null        null         0													

                                                    till          3         31    0.0     fldcult        null     0.000
                                                    fert          4         16    0.0      elem_n   broadcast    38.300	
                                                    plnt          5          5    0.0        corn        null         0
                                                    fert          6         11    0.0      elem_n   broadcast   123.200												

                                                    hvkl          9          5    0.0        corn       grain     0.000
                                                    till          9         10    0.0    chisplow        null     0.000												

                                                    fert         12          2    0.0    11_52_00   broadcast   142.700											

                                                    skip          0          0      0        null        null         0
mgt_02                              18           1
                                                irr_opt_sw_unlim
                                                    plnt          4         15    0.0       hay          null         0
                                                    fert          4         20    0.0      elem_n   broadcast       75.000
                                                    fert          4         20    0.0      elem_p   broadcast       50.000
                                                    harv          6         15    0.0        hay    hay_cut_low     0.000

                                                    fert          7          1    0.0      elem_n   broadcast       75.000
                                                    harv          7         25    0.0        hay    hay_cut_low     0.000

                                                    fert          8         15    0.0      elem_n   broadcast       75.000
                                                    harv         10          5    0.0        hay    hay_cut_high    0.000

                                                    skip          0          0    0.0        null        null         0

                                                    plnt          4         15    0.0       hay          null         0
                                                    fert          4         20    0.0      elem_n   broadcast       75.000
                                                    fert          4         20    0.0      elem_p   broadcast       50.000
                                                    harv          6         15    0.0        hay    hay_cut_low     0.000

                                                    fert          7          1    0.0      elem_n   broadcast       75.000
                                                    harv          7         25    0.0        hay    hay_cut_low     0.000

                                                    fert          8         15    0.0      elem_n   broadcast       75.000
                                                    harv         10          5    0.0        hay    hay_cut_high    0.000

                                                    skip          0          0    0.0        null   null            0
mgt_03                              20      0
                                                    plnt         3        1     0.0       past         null         0   # Plant pasture in March
                                                    graz         3       15     0.0     beef_low       null        30   # 30 days of grazing
                                                    fert         4        1     0.0      elem_n   broadcast      50.000  # Post-grazing fertilization

                                                    graz         5       15     0.0     beef_low       null        30   # Grazing second cycle
                                                    fert         6        1     0.0      elem_n   broadcast      50.000  # Post-grazing fertilization

                                                    graz         7        1     0.0     beef_low       null        30   # Grazing third cycle
                                                    fert         8        1     0.0      elem_n   broadcast      50.000  # Post-grazing fertilization

                                                    graz         9       15     0.0     beef_low       null        30   # Fourth grazing cycle
                                                    fert         10       1     0.0      elem_n   broadcast      50.000  # Post-grazing fertilization

                                                    skip          0        0     0.0        null        null         0   # End of year
                                                    plnt         3        1     0.0       past         null         0    # Re-plant pasture for next year
                                                    graz         3       15     0.0     beef_low       null        30
                                                    fert         4        1     0.0      elem_n   broadcast      50.000
                                                    
                                                    graz         5       15     0.0     beef_low       null        30
                                                    fert         6        1     0.0      elem_n   broadcast      50.000

                                                    graz         7        1     0.0     beef_low       null        30
                                                    fert         8        1     0.0      elem_n   broadcast      50.000

                                                    graz         9       15     0.0     beef_low       null        30
                                                    fert         10       1     0.0      elem_n   broadcast      50.000

                                                    skip          0        0     0.0        null        null         0   # End of second year
