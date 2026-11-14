%main
% alg 1 non coop lin
% alg 2 non coop particles
% alg 3 coop range linearized
% alg 4 coop range particles
% alg 5 coop lidar
% alg 6 coop std belief propagation
% alg 7 ekf distributed
% alg 8 joao gaspar

R = run_simulation({@alg_pred_lin, @alg_spawn_lin, @alg_pred_pf, @alg_spawn_pf}, 10);