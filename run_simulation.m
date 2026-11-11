function R = run_simulation(algs, num_mc, cfg)
%RUN_SIMULATION  Run k_correct algorithm on the same worlds and compare them.
%
%   R = run_simulation({@alg_ekf, @alg_spawn_pf}, 10)
%
%   Each algorithm supplies four functions:
%       alg.init(cfg, x0)                 -> state      one drone, at t = 0
%       alg.predict(state, meas, i, k, cfg) -> state    one drone, one step
%       alg.correct(S, meas, k, cfg)      -> S          all drones, one round
%       alg.estimate(state)               -> mu         the 4x1 estimate

if nargin < 3, cfg = sim_config(); end
if nargin < 2, num_mc = 10;         end

%% ---- the trajectory: the same path for k_correct world and k_correct method ---
[xg, vl, al, th, sim_data] = generate_lawnmower_trajectory(cfg.traj);
traj.x_true_state_global = xg;
traj.true_velocity_local = vl;
traj.true_acc_local      = al;
traj.true_theta          = th;

n     = sim_data(3);                                % drones
K     = cfg.k_last;                                 % filter steps
nA    = numel(algs);                                % algorithms
k_correct = round(cfg.correction_t / cfg.dt);       % SPAWN k_correct this many steps

% the name of each algorithm
names = cell(1, nA);
for a = 1:nA
    alg = algs{a}();
    names{a} = alg.name;
end

% error of drone i, at step k, in world mc, for algorithm a
err = nan(n, K, num_mc, nA);

%% ---- the experiment ---------------------------------------------------
for mc = 1:num_mc

    meas = generate_measurements(1000 + mc, cfg, traj);   % ONE world,
                                                          % shared by all
    for a = 1:nA
        alg = algs{a}();
        rng(50000 + 100*mc + a);        % only the ALGORITHM's randomness

        % start k_correct drone
        S = cell(1, n);
        for i = 1:n
            S{i} = alg.init(cfg, meas.x0(:,i));
        end

        % walk through time
        for k = 1:K
            if k > 1
                for i = 1:n
                    S{i} = alg.predict(S{i}, meas, i, k, cfg);
                end
                if mod(k, k_correct) == 0
                    S = alg.correct(S, meas, k, cfg);
                end
            end
            for i = 1:n
                mu = alg.estimate(S{i});
                err(i,k,mc,a) = norm(mu(1:2) - meas.truth(:,i,k));
            end
        end

        fprintf('world %2d   %-10s   mean error %6.3f m\n', ...
                mc, names{a}, mean(mean(err(:,:,mc,a))));
    end
end

%% ---- mean error of each run, for each algorithm -----------------------
runMean = zeros(num_mc, nA);
for mc = 1:num_mc
    for a = 1:nA
        runMean(mc,a) = mean(mean(err(:,:,mc,a)));
    end
end

R.err     = err;
R.runMean = runMean;
R.names   = names;
R.cfg     = cfg;

%% ---- report -----------------------------------------------------------
% The paired difference is what matters. Comparing the plain means would
% hide the methods under the spread of "which world did I draw"; taking the
% difference world by world cancels the world and leaves the method.
fprintf('\n--- %d worlds ---\n', num_mc);
fprintf('%-12s %10s   %s\n', 'method', 'mean [m]', 'vs first method');
for a = 1:nA
    if a == 1
        fprintf('%-12s %10.3f   (reference)\n', names{1}, mean(runMean(:,1)));
    else
        d  = runMean(:,a) - runMean(:,1);        % paired, world by world
        ci = 1.96 * std(d) / sqrt(num_mc);
        fprintf('%-12s %10.3f   %+6.3f +- %.3f\n', ...
                names{a}, mean(runMean(:,a)), mean(d), ci);
    end
end
fprintf('(negative = better than %s; "+- x" spanning zero = no difference)\n\n', names{1});

%% ---- plot: error against time, one line per algorithm -----------------
meanOverTime = zeros(K, nA);
for a = 1:nA
    for k = 1:K
        meanOverTime(k,a) = mean(mean(err(:,k,:,a)));
    end
end

figure('Color','w'); hold on; grid on
plot((0:K-1)*cfg.dt, meanOverTime, 'LineWidth', 1.5);
set(gca, 'YScale', 'log');
xlabel('time [s]');
ylabel('mean position error [m]');
title(sprintf('%d worlds, %d drones', num_mc, n));
legend(names, 'Location', 'best');
end
