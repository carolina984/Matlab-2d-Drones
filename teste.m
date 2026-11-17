%main
% alg 1 non coop lin
% alg 2 non coop particles
% alg 3 coop range linearized
% alg 4 coop range particles
% alg 5 coop lidar
% alg 6 coop std belief propagation
% alg 7 ekf distributed
% alg 8 joao gaspar

%R1 = run_simulation({@alg_pred_lin, @alg_spawn_lin, @alg_pred_pf, @alg_spawn_pf}, 10, sim_config(struct('num_spawn_iter',1)));

%R2 = run_simulation({@alg_pred_lin, @alg_spawn_lin, @alg_pred_pf, @alg_spawn_pf}, 10, sim_config(struct('num_spawn_iter',3)));
%R1 = run_simulation({@alg_pred_lin, @alg_spawn_lin, @alg_pred_pf, @alg_spawn_pf}, 5, sim_config(struct('num_spawn_iter',1)));
%R1 = run_simulation({@alg_pred_lin, @alg_spawn_lin}, 10, sim_config(struct('num_spawn_iter',1)));
% R2 = run_simulation({@alg_pred_lin, @alg_spawn_lin}, 10, sim_config(struct('num_spawn_iter',2)));
% R3 = run_simulation({@alg_pred_lin, @alg_spawn_lin}, 10, sim_config(struct('num_spawn_iter',3)));
R4 = run_simulation({@alg_pred_lin, @alg_spawn_range_lin, @alg_spawn_lidar}, 10, sim_config(struct('num_spawn_iter',4)));
% % 
% % for a = [0.05 0.2 0.5 1.0]
% %     cfg = sim_config(struct('gps_availability', a));
% %     run_simulation({ alg_with(@alg_spawn_lin,'num_spawn_iter',1), ...
% %                      alg_with(@alg_spawn_lin,'num_spawn_iter',2), ...
% %                      alg_with(@alg_spawn_lin,'num_spawn_iter',4) }, 5, cfg);
% % end