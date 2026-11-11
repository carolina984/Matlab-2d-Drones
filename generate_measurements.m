function meas = generate_measurements(seed, cfg, traj)
%GENERATE_MEASUREMENTS  World + measurements for one Monte Carlo run.
%
%   Everything a sensor would have produced, drawn once from `seed` and
%   shared by every algorithm. No algorithm may call randn: if it draws a
%   measurement, it belongs here.
%
%   meas.theta (n,K)      heading measurement [rad]
%   meas.u     (2,n,K)    body-frame acceleration
%   meas.vel   (2,n,K)    body-frame velocity
%   meas.gps   (2,n,K)    GPS position
%   meas.gps_on(n,K)      logical, is there a fix
%   meas.z     (n,n,K)    range measurement, NaN if out of range
%   meas.x0    (4,n)      initial state handed to the filters
%   meas.truth (2,n,K)    true position -- FOR SCORING ONLY, never to a filter
%   meas.index (1,K)      trajectory sample index per filter step

rng(seed);

n = cfg.num_uavs;
K = cfg.k_last;

meas.index = floor((0:K-1)*cfg.dt / cfg.dt_sim) + 1;

% ---- initial state error ----------------------------------------------
meas.x0 = zeros(4, n);
for i = 1:n
    meas.x0(:,i) = traj.x_true_state_global{i}(1:4, meas.index(1)) ...
                   + mvnrnd(zeros(4,1), cfg.P_0)';
end

% ---- per-drone, per-step measurements ----------------------------------
meas.theta = zeros(n, K);
meas.u     = zeros(2, n, K);
meas.vel   = zeros(2, n, K);
meas.gps   = zeros(2, n, K);
meas.truth = zeros(2, n, K);

for i = 1:n
    idx = meas.index;
    meas.theta(i,:)   = traj.true_theta{i}(idx)              + cfg.sigma_theta*randn(1,K);
    meas.u(:,i,:)     = traj.true_acc_local{i}(1:2,idx)      + mvnrnd([0 0], cfg.A_noise, K).';
    meas.vel(:,i,:)   = traj.true_velocity_local{i}(1:2,idx) + mvnrnd([0 0], cfg.V_cov,   K).';
    meas.gps(:,i,:)   = traj.x_true_state_global{i}(1:2,idx) + mvnrnd([0 0], cfg.gps_cov, K).';
    meas.truth(:,i,:) = traj.x_true_state_global{i}(1:2,idx);
end

% ---- GPS availability (simplified Gilbert) -----------------------------
meas.gps_on = generate_gps_availability(n, K, cfg.gps_availability, ...
                                        cfg.gps_mean_outage);
meas.gps_on(setdiff(1:n, cfg.gps_receivers), :) = false;

% ---- range measurements---------------
meas.z_range = nan(n, n, K);
for k = 1:K
    for i = 1:n
        for j = 1:n
            if j == i, continue; end
            d = norm(meas.truth(:,j,k) - meas.truth(:,i,k));
            if d < cfg.range_max
                meas.z_range(i, j, k) = d + normrnd(0, cfg.range_std);
            end
        end
    end
end

% ---- LiDAR measurements---------------
meas.z_lidar = nan(2, n, n, K);
for k = 1:K
    for i = 1:n
        th = traj.true_theta{i}(meas.index(k));        % the TRUE heading
        R  = [cos(th) -sin(th); sin(th) cos(th)];      % global -> body
        for j = 1:n
            if j == i, continue; end
            d = meas.truth(:,j,k) - meas.truth(:,i,k);
            if norm(d) < cfg.range_max
                meas.z_lidar(:,i,j,k) = R*d + cfg.lidar_std*randn(2,1);
            end
        end
    end
end

meas.seed = seed;
end
