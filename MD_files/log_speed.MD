 Device : cuda  |  Board: 5x5  |  Repeats: 10 (+3 warm-up)
  Configs: 8  |  SIM counts: [0, 50, 100, 200, 500, 1000]  |  Stages: 3

  Stage 'first_move': 0/60 lines drawn (0% filled), 60 valid moves remaining
  Stage 'mid_game': 24/60 lines drawn (40% filled), 36 valid moves remaining
  Stage 'end_game': 48/60 lines drawn (80% filled), 12 valid moves remaining


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 — Pure NN Inference Speed  (raw nnet.predict() calls, no MCTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Config                Params     ms/call        ±   calls/sec
  ------------------------------------------------------------
  64ch x3blk           238,758       1.511  ± 0.089       662.0 calls/s
  64ch x5blk           386,726       1.928  ± 0.301       518.6 calls/s
  128ch x3blk          905,766       1.498  ± 0.083       667.3 calls/s
  128ch x5blk        1,496,614       1.925  ± 0.060       519.4 calls/s
  128ch x8blk        2,382,886       2.573  ± 0.137       388.7 calls/s
  192ch x5blk        3,343,782       2.051  ± 0.294       487.7 calls/s
  256ch x5blk        5,928,230       1.929  ± 0.279       518.5 calls/s
  256ch x10blk      11,831,590       3.042  ± 0.166       328.7 calls/s

  Fastest: 128ch x3blk  (1.498 ms/call)
  Slowest: 256ch x10blk  (3.042 ms/call)
  Inference speed-up (fastest vs slowest): 2.03×

  STEP 2 — MCTS Grid Search  (primary metric: ms / NN-call)
  NOTE: cells marked [SAT] mean the tree is exhausted
  (nn_calls < 80% of n_sims). Those timings measure re-traversal,
  NOT inference speed, and should be IGNORED for model comparison.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 64ch x3blk  |  Parameters: 238,758
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [  0.7%]       sims=   0 | first_move     |     3.66 ms (+-0.23) | ms/NN_call=3.6638 | NN_calls~0/0
  [  1.4%]       sims=   0 | mid_game       |     4.85 ms (+-1.57) | ms/NN_call=4.8450 | NN_calls~0/0
  [  2.1%]       sims=   0 | end_game       |     6.57 ms (+-1.50) | ms/NN_call=6.5660 | NN_calls~0/0
  [  2.8%]       sims=  50 | first_move     |   219.28 ms (+-52.52) | ms/NN_call=4.3856 | NN_calls~50/50
  [  3.5%]       sims=  50 | mid_game       |   247.73 ms (+-56.60) | ms/NN_call=4.9547 | NN_calls~50/50
  [  4.2%]       sims=  50 | end_game       |   248.99 ms (+-78.63) | ms/NN_call=4.9798 | NN_calls~50/50
  [  4.9%]       sims= 100 | first_move     |   496.15 ms (+-78.77) | ms/NN_call=4.9615 | NN_calls~100/100
  [  5.6%]       sims= 100 | mid_game       |   479.83 ms (+-59.22) | ms/NN_call=4.7983 | NN_calls~100/100
  [  6.2%]       sims= 100 | end_game       |   465.79 ms (+-60.17) | ms/NN_call=5.0085 | NN_calls~93/100
  [  6.9%]       sims= 200 | first_move     |   988.27 ms (+-42.06) | ms/NN_call=4.9413 | NN_calls~200/200
  [  7.6%]       sims= 200 | mid_game       |   995.75 ms (+-42.49) | ms/NN_call=4.9787 | NN_calls~200/200
  [  8.3%]       sims= 200 | end_game       |   904.57 ms (+-33.35) | ms/NN_call=5.0535 | NN_calls~179/200
  [  9.0%]       sims= 500 | first_move     |  2523.79 ms (+-109.94) | ms/NN_call=5.0476 | NN_calls~500/500
  [  9.7%]       sims= 500 | mid_game       |  2499.79 ms (+-72.50) | ms/NN_call=4.9996 | NN_calls~500/500
  [ 10.4%]       sims= 500 | end_game       |  2176.88 ms (+-54.99) | ms/NN_call=5.2709 | NN_calls~413/500
  [ 11.1%]       sims=1000 | first_move     |  5322.15 ms (+-142.86) | ms/NN_call=5.3222 | NN_calls~1000/1000
  [ 11.8%]       sims=1000 | mid_game       |  5104.82 ms (+-91.00) | ms/NN_call=5.1048 | NN_calls~1000/1000
  [ 12.5%]       sims=1000 | end_game       |  4410.15 ms (+-76.14) | ms/NN_call=5.4112 | NN_calls~815/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 64ch x5blk  |  Parameters: 386,726
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 13.2%]       sims=   0 | first_move     |     6.37 ms (+-2.44) | ms/NN_call=6.3663 | NN_calls~0/0
  [ 13.9%]       sims=   0 | mid_game       |     5.62 ms (+-1.37) | ms/NN_call=5.6169 | NN_calls~0/0
  [ 14.6%]       sims=   0 | end_game       |     5.25 ms (+-1.85) | ms/NN_call=5.2474 | NN_calls~0/0
  [ 15.3%]       sims=  50 | first_move     |   285.56 ms (+-125.58) | ms/NN_call=5.7111 | NN_calls~50/50
  [ 16.0%]       sims=  50 | mid_game       |   248.00 ms (+-62.31) | ms/NN_call=4.9599 | NN_calls~50/50
  [ 16.7%]       sims=  50 | end_game       |   252.77 ms (+-51.75) | ms/NN_call=5.0554 | NN_calls~50/50
  [ 17.4%]       sims= 100 | first_move     |   517.57 ms (+-54.34) | ms/NN_call=5.1757 | NN_calls~100/100
  [ 18.1%]       sims= 100 | mid_game       |   518.60 ms (+-59.97) | ms/NN_call=5.1860 | NN_calls~100/100
  [ 18.8%]       sims= 100 | end_game       |   524.87 ms (+-60.89) | ms/NN_call=5.3559 | NN_calls~98/100
  [ 19.4%]       sims= 200 | first_move     |  1076.48 ms (+-53.21) | ms/NN_call=5.3824 | NN_calls~200/200
  [ 20.1%]       sims= 200 | mid_game       |  1068.45 ms (+-59.23) | ms/NN_call=5.3422 | NN_calls~200/200
  [ 20.8%]       sims= 200 | end_game       |  1020.60 ms (+-43.00) | ms/NN_call=5.3434 | NN_calls~191/200
  [ 21.5%]       sims= 500 | first_move     |  2771.89 ms (+-66.55) | ms/NN_call=5.5438 | NN_calls~500/500
  [ 22.2%]       sims= 500 | mid_game       |  2722.00 ms (+-92.76) | ms/NN_call=5.4440 | NN_calls~500/500
  [ 22.9%]       sims= 500 | end_game       |  2601.92 ms (+-57.13) | ms/NN_call=5.6563 | NN_calls~460/500
  [ 23.6%]       sims=1000 | first_move     |  5780.53 ms (+-170.73) | ms/NN_call=5.7805 | NN_calls~1000/1000
  [ 24.3%]       sims=1000 | mid_game       |  5681.72 ms (+-497.10) | ms/NN_call=5.6817 | NN_calls~1000/1000
  [ 25.0%]       sims=1000 | end_game       |  5147.14 ms (+-84.10) | ms/NN_call=5.6875 | NN_calls~905/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 128ch x3blk  |  Parameters: 905,766
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 25.7%]       sims=   0 | first_move     |     5.72 ms (+-1.67) | ms/NN_call=5.7155 | NN_calls~0/0
  [ 26.4%]       sims=   0 | mid_game       |     5.02 ms (+-1.22) | ms/NN_call=5.0179 | NN_calls~0/0
  [ 27.1%]       sims=   0 | end_game       |     4.99 ms (+-1.29) | ms/NN_call=4.9947 | NN_calls~0/0
  [ 27.8%]       sims=  50 | first_move     |   263.42 ms (+-54.75) | ms/NN_call=5.2685 | NN_calls~50/50
  [ 28.5%]       sims=  50 | mid_game       |   256.91 ms (+-56.86) | ms/NN_call=5.1383 | NN_calls~50/50
  [ 29.2%]       sims=  50 | end_game       |   250.91 ms (+-55.28) | ms/NN_call=5.0182 | NN_calls~50/50
  [ 29.9%]       sims= 100 | first_move     |   534.65 ms (+-55.34) | ms/NN_call=5.3465 | NN_calls~100/100
  [ 30.6%]       sims= 100 | mid_game       |   538.02 ms (+-78.98) | ms/NN_call=5.3802 | NN_calls~100/100
  [ 31.2%]       sims= 100 | end_game       |   497.68 ms (+-59.15) | ms/NN_call=5.2387 | NN_calls~95/100
  [ 31.9%]       sims= 200 | first_move     |  1093.98 ms (+-53.40) | ms/NN_call=5.4699 | NN_calls~200/200
  [ 32.6%]       sims= 200 | mid_game       |  1080.27 ms (+-46.61) | ms/NN_call=5.4014 | NN_calls~200/200
  [ 33.3%]       sims= 200 | end_game       |   974.68 ms (+-16.66) | ms/NN_call=5.4149 | NN_calls~180/200
  [ 34.0%]       sims= 500 | first_move     |  2769.74 ms (+-89.21) | ms/NN_call=5.5395 | NN_calls~500/500
  [ 34.7%]       sims= 500 | mid_game       |  2747.25 ms (+-78.37) | ms/NN_call=5.4945 | NN_calls~500/500
  [ 35.4%]       sims= 500 | end_game       |  2520.45 ms (+-46.46) | ms/NN_call=5.6010 | NN_calls~450/500
  [ 36.1%]       sims=1000 | first_move     |  5901.15 ms (+-196.56) | ms/NN_call=5.9012 | NN_calls~1000/1000
  [ 36.8%]       sims=1000 | mid_game       |  5535.22 ms (+-155.71) | ms/NN_call=5.5352 | NN_calls~1000/1000
  [ 37.5%]       sims=1000 | end_game       |  5030.18 ms (+-144.15) | ms/NN_call=5.5582 | NN_calls~905/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 128ch x5blk  |  Parameters: 1,496,614
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 38.2%]       sims=   0 | first_move     |     7.00 ms (+-2.11) | ms/NN_call=7.0040 | NN_calls~0/0
  [ 38.9%]       sims=   0 | mid_game       |     5.92 ms (+-1.19) | ms/NN_call=5.9193 | NN_calls~0/0
  [ 39.6%]       sims=   0 | end_game       |     5.19 ms (+-0.63) | ms/NN_call=5.1880 | NN_calls~0/0
  [ 40.3%]       sims=  50 | first_move     |   262.06 ms (+-56.20) | ms/NN_call=5.2412 | NN_calls~50/50
  [ 41.0%]       sims=  50 | mid_game       |   261.09 ms (+-52.85) | ms/NN_call=5.2218 | NN_calls~50/50
  [ 41.7%]       sims=  50 | end_game       |   263.28 ms (+-53.08) | ms/NN_call=5.2656 | NN_calls~50/50
  [ 42.4%]       sims= 100 | first_move     |   521.34 ms (+-65.67) | ms/NN_call=5.2134 | NN_calls~100/100
  [ 43.1%]       sims= 100 | mid_game       |   533.55 ms (+-56.40) | ms/NN_call=5.3355 | NN_calls~100/100
  [ 43.8%]       sims= 100 | end_game       |   497.98 ms (+-50.26) | ms/NN_call=5.4128 | NN_calls~92/100
  [ 44.4%]       sims= 200 | first_move     |  1074.38 ms (+-42.00) | ms/NN_call=5.3719 | NN_calls~200/200
  [ 45.1%]       sims= 200 | mid_game       |  1075.96 ms (+-55.94) | ms/NN_call=5.3798 | NN_calls~200/200
  [ 45.8%]       sims= 200 | end_game       |   997.11 ms (+-38.22) | ms/NN_call=5.4487 | NN_calls~183/200
  [ 46.5%]       sims= 500 | first_move     |  2762.82 ms (+-59.38) | ms/NN_call=5.5256 | NN_calls~500/500
  [ 47.2%]       sims= 500 | mid_game       |  2756.03 ms (+-76.66) | ms/NN_call=5.5121 | NN_calls~500/500
  [ 47.9%]       sims= 500 | end_game       |  2545.79 ms (+-62.36) | ms/NN_call=5.6198 | NN_calls~453/500
  [ 48.6%]       sims=1000 | first_move     |  5801.94 ms (+-106.63) | ms/NN_call=5.8019 | NN_calls~1000/1000
  [ 49.3%]       sims=1000 | mid_game       |  5461.09 ms (+-84.42) | ms/NN_call=5.4611 | NN_calls~1000/1000
  [ 50.0%]       sims=1000 | end_game       |  5071.37 ms (+-124.31) | ms/NN_call=5.7564 | NN_calls~881/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 128ch x8blk  |  Parameters: 2,382,886
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 50.7%]       sims=   0 | first_move     |     5.59 ms (+-0.98) | ms/NN_call=5.5931 | NN_calls~0/0
  [ 51.4%]       sims=   0 | mid_game       |     5.13 ms (+-0.20) | ms/NN_call=5.1252 | NN_calls~0/0
  [ 52.1%]       sims=   0 | end_game       |     5.16 ms (+-0.19) | ms/NN_call=5.1557 | NN_calls~0/0
  [ 52.8%]       sims=  50 | first_move     |   279.49 ms (+-55.05) | ms/NN_call=5.5899 | NN_calls~50/50
  [ 53.5%]       sims=  50 | mid_game       |   285.42 ms (+-56.40) | ms/NN_call=5.7084 | NN_calls~50/50
  [ 54.2%]       sims=  50 | end_game       |   256.18 ms (+-54.67) | ms/NN_call=5.5692 | NN_calls~46/50
  [ 54.9%]       sims= 100 | first_move     |   552.57 ms (+-57.33) | ms/NN_call=5.5257 | NN_calls~100/100
  [ 55.6%]       sims= 100 | mid_game       |   556.04 ms (+-56.42) | ms/NN_call=5.5604 | NN_calls~100/100
  [ 56.2%]       sims= 100 | end_game       |   515.61 ms (+-62.92) | ms/NN_call=5.7933 | NN_calls~89/100
  [ 56.9%]       sims= 200 | first_move     |  1139.10 ms (+-42.10) | ms/NN_call=5.6955 | NN_calls~200/200
  [ 57.6%]       sims= 200 | mid_game       |  1137.68 ms (+-50.67) | ms/NN_call=5.6884 | NN_calls~200/200
  [ 58.3%]       sims= 200 | end_game       |  1052.39 ms (+-55.49) | ms/NN_call=5.8143 | NN_calls~181/200
  [ 59.0%]       sims= 500 | first_move     |  2935.76 ms (+-83.61) | ms/NN_call=5.8715 | NN_calls~500/500
  [ 59.7%]       sims= 500 | mid_game       |  2911.52 ms (+-70.06) | ms/NN_call=5.8230 | NN_calls~500/500
  [ 60.4%]       sims= 500 | end_game       |  2558.75 ms (+-50.05) | ms/NN_call=5.9784 | NN_calls~428/500
  [ 61.1%]       sims=1000 | first_move     |  6106.19 ms (+-161.90) | ms/NN_call=6.1062 | NN_calls~1000/1000
  [ 61.8%]       sims=1000 | mid_game       |  5845.48 ms (+-120.21) | ms/NN_call=5.8455 | NN_calls~1000/1000
  [ 62.5%]       sims=1000 | end_game       |  5086.24 ms (+-102.16) | ms/NN_call=6.0335 | NN_calls~843/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 192ch x5blk  |  Parameters: 3,343,782
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 63.2%]       sims=   0 | first_move     |     5.16 ms (+-0.86) | ms/NN_call=5.1575 | NN_calls~0/0
  [ 63.9%]       sims=   0 | mid_game       |     4.99 ms (+-0.54) | ms/NN_call=4.9854 | NN_calls~0/0
  [ 64.6%]       sims=   0 | end_game       |     5.08 ms (+-0.61) | ms/NN_call=5.0801 | NN_calls~0/0
  [ 65.3%]       sims=  50 | first_move     |   296.25 ms (+-128.50) | ms/NN_call=5.9250 | NN_calls~50/50
  [ 66.0%]       sims=  50 | mid_game       |   255.85 ms (+-59.00) | ms/NN_call=5.1170 | NN_calls~50/50
  [ 66.7%]       sims=  50 | end_game       |   270.38 ms (+-60.48) | ms/NN_call=5.4076 | NN_calls~50/50
  [ 67.4%]       sims= 100 | first_move     |   533.27 ms (+-61.80) | ms/NN_call=5.3327 | NN_calls~100/100
  [ 68.1%]       sims= 100 | mid_game       |   551.32 ms (+-50.94) | ms/NN_call=5.5132 | NN_calls~100/100
  [ 68.8%]       sims= 100 | end_game       |   540.52 ms (+-57.94) | ms/NN_call=5.4052 | NN_calls~100/100
  [ 69.4%]       sims= 200 | first_move     |  1109.59 ms (+-44.27) | ms/NN_call=5.5479 | NN_calls~200/200
  [ 70.1%]       sims= 200 | mid_game       |  1095.05 ms (+-64.84) | ms/NN_call=5.4753 | NN_calls~200/200
  [ 70.8%]       sims= 200 | end_game       |  1041.64 ms (+-73.84) | ms/NN_call=5.2082 | NN_calls~200/200
  [ 71.5%]       sims= 500 | first_move     |  2761.41 ms (+-137.90) | ms/NN_call=5.5228 | NN_calls~500/500
  [ 72.2%]       sims= 500 | mid_game       |  2798.61 ms (+-76.36) | ms/NN_call=5.5972 | NN_calls~500/500
  [ 72.9%]       sims= 500 | end_game       |  2755.35 ms (+-88.74) | ms/NN_call=5.5107 | NN_calls~500/500
  [ 73.6%]       sims=1000 | first_move     |  5950.53 ms (+-165.28) | ms/NN_call=5.9505 | NN_calls~1000/1000
  [ 74.3%]       sims=1000 | mid_game       |  5585.15 ms (+-179.84) | ms/NN_call=5.5852 | NN_calls~1000/1000
  [ 75.0%]       sims=1000 | end_game       |  5557.73 ms (+-160.53) | ms/NN_call=5.5577 | NN_calls~1000/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 256ch x5blk  |  Parameters: 5,928,230
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 75.7%]       sims=   0 | first_move     |     6.43 ms (+-0.60) | ms/NN_call=6.4328 | NN_calls~0/0
  [ 76.4%]       sims=   0 | mid_game       |     6.54 ms (+-0.54) | ms/NN_call=6.5432 | NN_calls~0/0
  [ 77.1%]       sims=   0 | end_game       |     6.51 ms (+-0.76) | ms/NN_call=6.5103 | NN_calls~0/0
  [ 77.8%]       sims=  50 | first_move     |   244.30 ms (+-60.79) | ms/NN_call=4.8861 | NN_calls~50/50
  [ 78.5%]       sims=  50 | mid_game       |   254.68 ms (+-53.12) | ms/NN_call=5.0937 | NN_calls~50/50
  [ 79.2%]       sims=  50 | end_game       |   263.28 ms (+-53.82) | ms/NN_call=5.2657 | NN_calls~50/50
  [ 79.9%]       sims= 100 | first_move     |   543.58 ms (+-79.67) | ms/NN_call=5.4358 | NN_calls~100/100
  [ 80.6%]       sims= 100 | mid_game       |   535.73 ms (+-66.88) | ms/NN_call=5.3573 | NN_calls~100/100
  [ 81.2%]       sims= 100 | end_game       |   528.34 ms (+-66.87) | ms/NN_call=5.2834 | NN_calls~100/100
  [ 81.9%]       sims= 200 | first_move     |  1061.01 ms (+-85.67) | ms/NN_call=5.3050 | NN_calls~200/200
  [ 82.6%]       sims= 200 | mid_game       |  1036.48 ms (+-71.14) | ms/NN_call=5.1824 | NN_calls~200/200
  [ 83.3%]       sims= 200 | end_game       |  1004.55 ms (+-59.96) | ms/NN_call=5.3434 | NN_calls~188/200
  [ 84.0%]       sims= 500 | first_move     |  2728.05 ms (+-80.35) | ms/NN_call=5.4561 | NN_calls~500/500
  [ 84.7%]       sims= 500 | mid_game       |  2687.97 ms (+-96.45) | ms/NN_call=5.3759 | NN_calls~500/500
  [ 85.4%]       sims= 500 | end_game       |  2490.77 ms (+-88.71) | ms/NN_call=5.4984 | NN_calls~453/500
  [ 86.1%]       sims=1000 | first_move     |  5538.68 ms (+-122.26) | ms/NN_call=5.5387 | NN_calls~1000/1000
  [ 86.8%]       sims=1000 | mid_game       |  5352.92 ms (+-121.72) | ms/NN_call=5.3529 | NN_calls~1000/1000
  [ 87.5%] [SAT] sims=1000 | end_game       |  4114.05 ms (+-76.74) | ms/NN_call=5.5595 | NN_calls~740/1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: 256ch x10blk  |  Parameters: 11,831,590
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ 88.2%]       sims=   0 | first_move     |    10.66 ms (+-2.42) | ms/NN_call=10.6569 | NN_calls~0/0
  [ 88.9%]       sims=   0 | mid_game       |     9.08 ms (+-1.48) | ms/NN_call=9.0802 | NN_calls~0/0
  [ 89.6%]       sims=   0 | end_game       |     5.38 ms (+-0.27) | ms/NN_call=5.3832 | NN_calls~0/0
  [ 90.3%]       sims=  50 | first_move     |   329.60 ms (+-122.18) | ms/NN_call=6.5920 | NN_calls~50/50
  [ 91.0%]       sims=  50 | mid_game       |   301.68 ms (+-51.91) | ms/NN_call=6.0337 | NN_calls~50/50
  [ 91.7%]       sims=  50 | end_game       |   292.63 ms (+-55.11) | ms/NN_call=6.0964 | NN_calls~48/50
  [ 92.4%]       sims= 100 | first_move     |   597.98 ms (+-60.01) | ms/NN_call=5.9798 | NN_calls~100/100
  [ 93.1%]       sims= 100 | mid_game       |   602.26 ms (+-63.90) | ms/NN_call=6.0226 | NN_calls~100/100
  [ 93.8%] [SAT] sims= 100 | end_game       |   493.03 ms (+-68.60) | ms/NN_call=6.4029 | NN_calls~77/100
  [ 94.4%]       sims= 200 | first_move     |  1215.76 ms (+-35.10) | ms/NN_call=6.0788 | NN_calls~200/200
  [ 95.1%]       sims= 200 | mid_game       |  1226.63 ms (+-52.31) | ms/NN_call=6.1332 | NN_calls~200/200
  [ 95.8%] [SAT] sims= 200 | end_game       |   931.48 ms (+-94.02) | ms/NN_call=6.2938 | NN_calls~148/200
  [ 96.5%]       sims= 500 | first_move     |  3073.33 ms (+-86.49) | ms/NN_call=6.1467 | NN_calls~500/500
  [ 97.2%]       sims= 500 | mid_game       |  3097.15 ms (+-62.03) | ms/NN_call=6.1943 | NN_calls~500/500
  [ 97.9%] [SAT] sims= 500 | end_game       |  1619.28 ms (+-72.56) | ms/NN_call=6.8324 | NN_calls~237/500
  [ 98.6%]       sims=1000 | first_move     |  6192.87 ms (+-106.72) | ms/NN_call=6.1929 | NN_calls~1000/1000
  [ 99.3%]       sims=1000 | mid_game       |  6198.91 ms (+-92.23) | ms/NN_call=6.1989 | NN_calls~1000/1000
  [100.0%] [SAT] sims=1000 | end_game       |  2194.19 ms (+-101.39) | ms/NN_call=6.8568 | NN_calls~320/1000


============================================================================
  STAGE: first_move  (0% lines filled)
  Primary metric: ms / NN-call  (= mean_ms / actual_nn_calls)
  [SAT] = tree exhausted; cell is invalid for speed comparison
============================================================================
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        3.66        4.39        4.96        4.94        5.05        5.32 ms/NN
64ch x5blk         386,726        6.37        5.71        5.18        5.38        5.54        5.78 ms/NN
128ch x3blk        905,766        5.72        5.27        5.35        5.47        5.54        5.90 ms/NN
128ch x5blk      1,496,614        7.00        5.24        5.21        5.37        5.53        5.80 ms/NN
128ch x8blk      2,382,886        5.59        5.59        5.53        5.70        5.87        6.11 ms/NN
192ch x5blk      3,343,782        5.16        5.93        5.33        5.55        5.52        5.95 ms/NN
256ch x5blk      5,928,230        6.43        4.89        5.44        5.31        5.46        5.54 ms/NN
256ch x10blk    11,831,590       10.66        6.59        5.98        6.08        6.15        6.19 ms/NN

  Speed-up vs 256ch x10blk  (non-[SAT] cells only):
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        2.91x        1.50x        1.21x        1.23x        1.22x        1.16x
64ch x5blk         386,726        1.67x        1.15x        1.16x        1.13x        1.11x        1.07x
128ch x3blk        905,766        1.86x        1.25x        1.12x        1.11x        1.11x        1.05x
128ch x5blk      1,496,614        1.52x        1.26x        1.15x        1.13x        1.11x        1.07x
128ch x8blk      2,382,886        1.91x        1.18x        1.08x        1.07x        1.05x        1.01x
192ch x5blk      3,343,782        2.07x        1.11x        1.12x        1.10x        1.11x        1.04x
256ch x5blk      5,928,230        1.66x        1.35x        1.10x        1.15x        1.13x        1.12x
256ch x10blk    11,831,590        1.00x        1.00x        1.00x        1.00x        1.00x        1.00x


============================================================================
  STAGE: mid_game  (40% lines filled)
  Primary metric: ms / NN-call  (= mean_ms / actual_nn_calls)
  [SAT] = tree exhausted; cell is invalid for speed comparison
============================================================================
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        4.85        4.95        4.80        4.98        5.00        5.10 ms/NN
64ch x5blk         386,726        5.62        4.96        5.19        5.34        5.44        5.68 ms/NN
128ch x3blk        905,766        5.02        5.14        5.38        5.40        5.49        5.54 ms/NN
128ch x5blk      1,496,614        5.92        5.22        5.34        5.38        5.51        5.46 ms/NN
128ch x8blk      2,382,886        5.13        5.71        5.56        5.69        5.82        5.85 ms/NN
192ch x5blk      3,343,782        4.99        5.12        5.51        5.48        5.60        5.59 ms/NN
256ch x5blk      5,928,230        6.54        5.09        5.36        5.18        5.38        5.35 ms/NN
256ch x10blk    11,831,590        9.08        6.03        6.02        6.13        6.19        6.20 ms/NN

  Speed-up vs 256ch x10blk  (non-[SAT] cells only):
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        1.87x        1.22x        1.26x        1.23x        1.24x        1.21x
64ch x5blk         386,726        1.62x        1.22x        1.16x        1.15x        1.14x        1.09x
128ch x3blk        905,766        1.81x        1.17x        1.12x        1.14x        1.13x        1.12x
128ch x5blk      1,496,614        1.53x        1.16x        1.13x        1.14x        1.12x        1.14x
128ch x8blk      2,382,886        1.77x        1.06x        1.08x        1.08x        1.06x        1.06x
192ch x5blk      3,343,782        1.82x        1.18x        1.09x        1.12x        1.11x        1.11x
256ch x5blk      5,928,230        1.39x        1.18x        1.12x        1.18x        1.15x        1.16x
256ch x10blk    11,831,590        1.00x        1.00x        1.00x        1.00x        1.00x        1.00x


============================================================================
  STAGE: end_game  (80% lines filled)
  Primary metric: ms / NN-call  (= mean_ms / actual_nn_calls)
  [SAT] = tree exhausted; cell is invalid for speed comparison
============================================================================
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        6.57        4.98        5.01        5.05        5.27        5.41 ms/NN
64ch x5blk         386,726        5.25        5.06        5.36        5.34        5.66        5.69 ms/NN
128ch x3blk        905,766        4.99        5.02        5.24        5.41        5.60        5.56 ms/NN
128ch x5blk      1,496,614        5.19        5.27        5.41        5.45        5.62        5.76 ms/NN
128ch x8blk      2,382,886        5.16        5.57        5.79        5.81        5.98        6.03 ms/NN
192ch x5blk      3,343,782        5.08        5.41        5.41        5.21        5.51        5.56 ms/NN
256ch x5blk      5,928,230        6.51        5.27        5.28        5.34        5.50       [SAT] ms/NN
256ch x10blk    11,831,590        5.38        6.10       [SAT]       [SAT]       [SAT]       [SAT] ms/NN

  Speed-up vs 256ch x10blk  (non-[SAT] cells only):
Config              Params          NN       50sim      100sim      200sim      500sim     1000sim
--------------------------------------------------------------------------------------------------
64ch x3blk         238,758        0.82x        1.22x       [SAT]       [SAT]       [SAT]       [SAT]
64ch x5blk         386,726        1.03x        1.21x       [SAT]       [SAT]       [SAT]       [SAT]
128ch x3blk        905,766        1.08x        1.21x       [SAT]       [SAT]       [SAT]       [SAT]
128ch x5blk      1,496,614        1.04x        1.16x       [SAT]       [SAT]       [SAT]       [SAT]
128ch x8blk      2,382,886        1.04x        1.09x       [SAT]       [SAT]       [SAT]       [SAT]
192ch x5blk      3,343,782        1.06x        1.13x       [SAT]       [SAT]       [SAT]       [SAT]
256ch x5blk      5,928,230        0.83x        1.16x       [SAT]       [SAT]       [SAT]       [SAT]
256ch x10blk    11,831,590        1.00x        1.00x       [SAT]       [SAT]       [SAT]       [SAT]


============================================================================
  ms / NN-call  (mean over valid, non-saturated cells across all 3 stages)
  This is the CORRECT metric for comparing model inference speed
============================================================================
Config              50 sims    100 sims    200 sims    500 sims   1000 sims
---------------------------------------------------------------------------
64ch x3blk           4.7734      4.9228      4.9912      5.1060      5.2794 ms/NN_call
64ch x5blk           5.2421      5.2392      5.3560      5.5480      5.7166 ms/NN_call
128ch x3blk          5.1416      5.3218      5.4287      5.5450      5.6649 ms/NN_call
128ch x5blk          5.2428      5.3206      5.4001      5.5525      5.6731 ms/NN_call
128ch x8blk          5.6225      5.6265      5.7328      5.8910      5.9951 ms/NN_call
192ch x5blk          5.4832      5.4170      5.4105      5.5436      5.6978 ms/NN_call
256ch x5blk          5.0818      5.3588      5.2769      5.4435      5.4458 ms/NN_call
256ch x10blk         6.2407      6.0012      6.1060      6.1705      6.1959 ms/NN_call


============================================================================
  Fastest config per (stage, n_sims)  — [SAT] cells excluded:
============================================================================
  first_move     | NN-only    | 64ch x3blk  (3.6638 ms)
  first_move     |  50 sims   | 64ch x3blk  (4.3856 ms/NN_call)
  first_move     | 100 sims   | 64ch x3blk  (4.9615 ms/NN_call)
  first_move     | 200 sims   | 64ch x3blk  (4.9413 ms/NN_call)
  first_move     | 500 sims   | 64ch x3blk  (5.0476 ms/NN_call)
  first_move     | 1000 sims  | 64ch x3blk  (5.3222 ms/NN_call)
  mid_game       | NN-only    | 64ch x3blk  (4.8450 ms)
  mid_game       |  50 sims   | 64ch x3blk  (4.9547 ms/NN_call)
  mid_game       | 100 sims   | 64ch x3blk  (4.7983 ms/NN_call)
  mid_game       | 200 sims   | 64ch x3blk  (4.9787 ms/NN_call)
  mid_game       | 500 sims   | 64ch x3blk  (4.9996 ms/NN_call)
  mid_game       | 1000 sims  | 64ch x3blk  (5.1048 ms/NN_call)
  end_game       | NN-only    | 128ch x3blk  (4.9947 ms)
  end_game       |  50 sims   | 64ch x3blk  (4.9798 ms/NN_call)
  end_game       | 100 sims   | 64ch x3blk  (5.0085 ms/NN_call)
  end_game       | 200 sims   | 64ch x3blk  (5.0535 ms/NN_call)
  end_game       | 500 sims   | 64ch x3blk  (5.2709 ms/NN_call)
  end_game       | 1000 sims  | 64ch x3blk  (5.4112 ms/NN_call)


============================================================================
  DEFINITIVE RANKING — Pure NN Inference Speed
  (nnet.predict() call, no MCTS overhead)
============================================================================
  #1  128ch x3blk           905,766 params     1.498 ms  ±0.083     667.3 calls/s  (2.03x vs slowest)
  #2  64ch x3blk            238,758 params     1.511 ms  ±0.089     662.0 calls/s  (2.01x vs slowest)
  #3  128ch x5blk         1,496,614 params     1.925 ms  ±0.060     519.4 calls/s  (1.58x vs slowest)
  #4  64ch x5blk            386,726 params     1.928 ms  ±0.301     518.6 calls/s  (1.58x vs slowest)
  #5  256ch x5blk         5,928,230 params     1.929 ms  ±0.279     518.5 calls/s  (1.58x vs slowest)
  #6  192ch x5blk         3,343,782 params     2.051 ms  ±0.294     487.7 calls/s  (1.48x vs slowest)
  #7  128ch x8blk         2,382,886 params     2.573 ms  ±0.137     388.7 calls/s  (1.18x vs slowest)
  #8  256ch x10blk       11,831,590 params     3.042 ms  ±0.166     328.7 calls/s  (1.00x vs slowest)

  Recommendation for fastest model with acceptable quality:
  → 128ch x3blk  (1.498 ms/call, 905,766 params)

Benchmark complete.
