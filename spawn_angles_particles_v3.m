% clear

%% Load trajectory
% load("true_acc_local.mat");
% load("true_acc_global.mat");
% load("x_true_state_global.mat");
% load("true_theta.mat");
% load("true_velocity_local.mat");
% load("sim_data_sim1.mat");


% The generator leaves the first and last few samples corrupt: minjerkpolytraj
% starts and ends at rest, so atan2(0,0) gives a bogus heading there, the
% formation is placed unrotated for one sample, and differentiating that jump
% writes thousands of m/s into column 1. Hold a clean interior sample.
for i = 1:numel(x_true_state_global)
    x_true_state_global{i}(:,1) = x_true_state_global{i}(:,10);
    true_velocity_local{i}(:,1) = true_velocity_local{i}(:,10);
    true_acc_local{i}(:,1)      = true_acc_local{i}(:,10);
    true_acc_global{i}(:,1)     = true_acc_global{i}(:,10);
    true_theta{i}(1)            = true_theta{i}(10);
end

%% Simulation selection
% [ measurement type , spawn iterations , particles , gps , start ]
%   0 none, 1 lidar, 2 range            0 no 1 yes   0 no 1 yes   0 warm 1 cold
spawn_parameters = [0, 1, 0, 1, 0];

k_last = 100;                    % last time step of the filter loop
num_mc = 1;                     % Monte Carlo runs

debug_steps = [4 8];               % which k to draw, e.g. [4 8]. [] = no figures.
                                % SPAWN only runs when rem(k, measurement_interval/dt) == 0,
                                % so with dt = 0.5 and interval 2 the live steps are 4, 8, 12...

% particles
inter_samples = 500;
intra_samples = 2000;
particle_h    = 0.4;
kk            = 2;              % candidate oversampling constant

% measurements
measurement_interval = 2;
velocity_std         = 0.2;

% range
range_d_max = 100;
range_std   = 0.4;

% lidar
lidar_d_max = 100;
lidar_std   = 0.1;

% gps
gps_interval = 0.5;
gps_std      = 0.00005;

%% Extract selection
measurement_type = spawn_parameters(1);
spawn_on         = measurement_type ~= 0;

if spawn_on
    num_spawn_iter = spawn_parameters(2);
    particles      = (measurement_type == 2) * spawn_parameters(3);
else
    num_spawn_iter = 0;
    particles      = 0;
end

gps_activated = spawn_parameters(4);
cold_start    = logical(spawn_parameters(5));

%% Simulation parameters
T        = sim_data(1);
dt_sim   = sim_data(2);         % trajectory sample time (s)
num_uavs = sim_data(3);

tvec_sim  = 0:dt_sim:T;
steps_sim = length(tvec_sim);

dt    = 0.5;                    % filter step (s)
tvec  = 0:dt:T;
steps = length(tvec);

% refuse to read past the end of the trajectory
k_max_legal = floor((size(x_true_state_global{1},2)-1)*dt_sim/dt) + 1;
if k_last > k_max_legal
    error('k_last = %d needs sample %d, but the trajectory has %d. Largest legal k is %d.', ...
        k_last, floor((k_last-1)*dt/dt_sim)+1, size(x_true_state_global{1},2), k_max_legal);
end

A = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];

B = [0.5*dt^2 0;
     0 0.5*dt^2;
     dt 0;
     0 dt];

H_vel = [0 0 1 0;
         0 0 0 1];

sigma_theta   = 0.1;
sigma_process = 0.5;           % motion model
sigma_a       = 0.02;           % noise added to acceleration

gps_cov = gps_std^2      * eye(2);
L_cov   = lidar_std^2    * eye(2);
V_cov   = velocity_std^2 * eye(2);
A_noise = sigma_a^2      * eye(2);
Q_cov   = sigma_process^2 * eye(4);

if cold_start
    p_pos_0 = 10.0;              % initial position sd [m]
else
    p_pos_0 = 1.0;
end
p_vel_0 = 0.1;                  % initial velocity sd [m/s]

P_0 = diag([p_pos_0^2, p_pos_0^2, p_vel_0^2, p_vel_0^2]);

%% Monte Carlo
mean_errors_mc           = zeros(num_mc, 1);
mean_errors_per_drone_mc = zeros(num_mc, num_uavs);

for mc = 1:num_mc
    rng(mc);

    % TODO paired seeds: draw the world noise (theta, acceleration, velocity,
    % gps, range) here, before the `if particles` branch, so both methods see
    % the same world. Right now gaussian_belief consumes the stream and the
    % two methods diverge from drone 2 onward.

    mu_pred        = cell(1, num_uavs);
    P_pred         = cell(1, num_uavs);
    theta_measured = cell(1, num_uavs);
    pos_error      = cell(1, num_uavs);
    mu_local       = cell(1, num_uavs);
    P_local        = cell(1, num_uavs);
    mu_spawn       = cell(1, num_uavs);
    p_spawn        = cell(1, num_uavs);
    lidar_points   = cell(1, num_uavs);
    bel_agents     = cell(1, num_uavs);
    broadcasted_message = cell(1, num_uavs);

    traceP   = zeros(num_uavs, steps);
    ess_pred = zeros(num_uavs, steps);

    %% Initial beliefs
    for i = 1:num_uavs
        mu_pred{i}   = zeros(4, steps);
        pos_error{i} = zeros(2, steps);

        e_0 = mvnrnd(zeros(4,1), P_0)';

        if particles
            bel_agents{i} = gaussian_belief(x_true_state_global{i}(1:4,1) + e_0, ...
                                            P_0, intra_samples);
            [mu0, P0] = belief_estimate(bel_agents{i});
            mu_pred{i}(:,1) = mu0(:);
            P_pred{i}       = P0;
        else
            mu_pred{i}(:,1) = x_true_state_global{i}(1:4,1) + e_0;
            P_pred{i}       = P_0;
        end

        traceP(i,1)       = trace(P_pred{i});
        pos_error{i}(:,1) = mu_pred{i}(1:2,1) - x_true_state_global{i}(1:2,1);
    end

    %% Time loop
    for k = 2:k_last
        index = floor((k-1)*dt / dt_sim) + 1;

        %% Prediction
        for i = 1:num_uavs
            theta_measured{i} = true_theta{i}(index) + sigma_theta * randn(1);

            u_measured   = true_acc_local{i}(1:2,index)      + mvnrnd([0 0], A_noise).';
            vel_measured = true_velocity_local{i}(1:2,index) + mvnrnd([0 0], V_cov).';
            gps_measured = x_true_state_global{i}(1:2,index) + mvnrnd([0 0], gps_cov).';

            add_gps_measure = gps_activated == 1 && i == 2 && ...
                              rem(k, round(gps_interval/dt)) == 0;

            if particles
                [bel_agents{i}, ess_pred(i,k)] = prediction_operation_particles( ...
                        bel_agents{i}, theta_measured{i}, sigma_theta, vel_measured, ...
                        u_measured, A, B, Q_cov, V_cov, gps_measured, add_gps_measure, gps_cov);

                [mu_pred{i}(:,k), P_pred{i}] = belief_estimate(bel_agents{i});
                broadcasted_message{i} = intrasamples_to_intersamples(bel_agents{i}, inter_samples);
            else
                [P_pred{i}, mu_pred{i}(:,k)] = prediction_operation_with_angles( ...
                        index, i, mu_pred{i}(:,k-1), P_pred{i}, theta_measured{i}, ...
                        sigma_theta, vel_measured, u_measured, A, B, Q_cov, H_vel, ...
                        V_cov, gps_measured, add_gps_measure);
            end

            traceP(i,k) = trace(P_pred{i});
        end

        %% SPAWN correction
        if spawn_on && rem(k, round(measurement_interval/dt)) == 0

            for i = 1:num_uavs
                mu_local{i} = mu_pred{i}(:,k);
                P_local{i}  = P_pred{i};
                mu_spawn{i} = mu_local{i};
                p_spawn{i}  = P_local{i};
            end

            switch measurement_type

                %% ---------------- LiDAR (Gaussian only) ----------------
                case 1
                    z_ij = cell(num_uavs, num_uavs);
                    for i = 1:num_uavs
                        for j = 1:num_uavs
                            if j == i, continue; end
                            d = x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index);
                            if norm(d) < lidar_d_max
                                th  = true_theta{i}(index);
                                rot = [cos(th) -sin(th); sin(th) cos(th)];
                                z_ij{i,j} = rot * (d + mvnrnd([0 0], L_cov)');
                            end
                        end
                    end

                    for f = 1:num_spawn_iter
                        mu_spawn_new = cell(1, num_uavs);
                        p_spawn_new  = cell(1, num_uavs);

                        for i = 1:num_uavs
                            mu_i = mu_local{i};
                            P_i  = P_local{i};

                            for j = 1:num_uavs
                                if j == i,                continue; end
                                if isempty(z_ij{i,j}),    continue; end

                                lidar_points{i} = [lidar_points{i}, mu_pred{i}(1:2,k-1)];

                                [mu_i, P_i] = intermeasurements_lidar(theta_measured{i}, ...
                                        sigma_theta, mu_spawn{j}, p_spawn{j}, mu_i, P_i, ...
                                        L_cov, z_ij{i,j});
                            end

                            mu_spawn_new{i} = mu_i;
                            p_spawn_new{i}  = P_i;
                        end

                        mu_spawn = mu_spawn_new;        % synchronous update
                        p_spawn  = p_spawn_new;
                    end

                %% ---------------- Range ----------------
                case 2
                    z_range = cell(num_uavs, num_uavs);
                    for i = 1:num_uavs
                        for j = 1:num_uavs
                            if j == i, continue; end
                            d = norm(x_true_state_global{j}(1:2,index) - ...
                                     x_true_state_global{i}(1:2,index));
                            if d < range_d_max
                                z_range{i,j} = d + normrnd(0, range_std);
                            end
                        end
                    end

                    if particles
                        pred_bel_agents = bel_agents;   % Alg. 3 line 2: b^(0) is the prediction

                        dbg.pred = bel_agents;          % snapshots: references only, no copy
                        dbg.msg  = cell(num_spawn_iter, num_uavs);
                        dbg.bel  = cell(1, num_spawn_iter);

                        for f = 1:num_spawn_iter
                            old_bel_agents = bel_agents;
                            old_broadcast  = broadcasted_message;   % frozen for this iteration
                            new_bel_agents = cell(1, num_uavs);
                            new_broadcast  = cell(1, num_uavs);

                            for i = 1:num_uavs
                                message_agent = {};
                                for j = 1:num_uavs
                                    if j == i,                      continue; end
                                    if isempty(z_range{i,j}),       continue; end
                                    if old_bel_agents{j}(1,6) == 0, continue; end   % no information

                                    message_agent{end+1} = compute_message(old_broadcast{j}, ...
                                            inter_samples, intra_samples, z_range{i,j}, range_std); %#ok<SAGROW>
                                end

                                [new_bel_agents{i}, new_broadcast{i}] = product_messages(f, ...
                                        pred_bel_agents{i}, message_agent, intra_samples, ...
                                        inter_samples, kk, intra_samples, particle_h);

                                dbg.msg{f,i} = message_agent;
                            end

                            bel_agents          = new_bel_agents;   % synchronous update
                            broadcasted_message = new_broadcast;
                            dbg.bel{f}          = bel_agents;
                        end

                        for i = 1:num_uavs
                            [mu_spawn{i}, p_spawn{i}] = belief_estimate(bel_agents{i});
                            mu_spawn{i} = mu_spawn{i}(:);   % belief_estimate returns a row
                        end

                        if ismember(k, debug_steps)
                            plot_spawn_debug(dbg, particle_h, k, tvec(k), ...
                                             x_true_state_global, index);
                        end

                    else
                        for f = 1:num_spawn_iter
                            mu_spawn_new = cell(1, num_uavs);
                            p_spawn_new  = cell(1, num_uavs);

                            for i = 1:num_uavs
                                mu_i = mu_local{i};
                                P_i  = P_local{i};

                                for j = 1:num_uavs
                                    if j == i,                 continue; end
                                    if isempty(z_range{i,j}),  continue; end

                                    [mu_i, P_i] = intermeasurement_range_linearized( ...
                                            mu_spawn{j}, p_spawn{j}, mu_i, P_i, ...
                                            range_std, z_range{i,j});
                                end

                                mu_spawn_new{i} = mu_i;
                                p_spawn_new{i}  = P_i;
                            end

                            mu_spawn = mu_spawn_new;    % synchronous update
                            p_spawn  = p_spawn_new;
                        end
                    end
            end

            for i = 1:num_uavs
                mu_pred{i}(:,k) = mu_spawn{i};
                P_pred{i}       = p_spawn{i};
                traceP(i,k)     = trace(P_pred{i});
            end
        end

        %% Error at this step
        for i = 1:num_uavs
            if particles
                mu__ = belief_estimate(bel_agents{i});
                pos_error{i}(:,k) = mu__(1:2).' - x_true_state_global{i}(1:2,index);
            else
                pos_error{i}(:,k) = mu_pred{i}(1:2,k) - x_true_state_global{i}(1:2,index);
            end
        end
    end

    for i = 1:num_uavs
        mean_errors_per_drone_mc(mc,i) = mean(vecnorm(pos_error{i}(:,1:k_last), 2, 1));
    end
    mean_errors_mc(mc) = mean(mean_errors_per_drone_mc(mc,:));
end

%% Trajectories
RGB = orderedcolors("gem");
figure; hold on; grid on;
leg = cell(1, 2*num_uavs);
for i = 1:num_uavs
    plot(x_true_state_global{i}(1,1:steps_sim), x_true_state_global{i}(2,1:steps_sim), ...
        'Color', RGB(mod(i-1,size(RGB,1))+1,:), 'LineStyle','--', 'LineWidth',2);
    plot(mu_pred{i}(1,1:k_last), mu_pred{i}(2,1:k_last), '-', ...
        'Color', RGB(mod(i-1,size(RGB,1))+1,:), 'LineWidth',2);
    leg{2*i-1} = sprintf('Ref %d', i);
    leg{2*i}   = sprintf('Est %d', i);
end
xlabel('x [m]'); ylabel('y [m]'); axis equal
legend(leg, 'Location','eastoutside');

figure
plot(tvec(1:k_last), traceP(:,1:k_last)'); grid on
xlabel('time [s]'); ylabel('trace(P)'); title('Covariance');

%% Simulation report
mean_error_per_drone = mean(mean_errors_per_drone_mc, 1);
mean_error_all       = mean(mean_errors_mc);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    SIMULATION REPORT\n');
fprintf('============================================================\n');
fprintf('Run date: %s\n', datetime("now"));
fprintf('Monte Carlo simulations: %d\n', num_mc);
fprintf('Time steps             : %d  (%.1f s)\n', k_last, (k_last-1)*dt);
fprintf('------------------------------------------------------------\n');

if cold_start
    fprintf('Initialization       : Cold start\n');
else
    fprintf('Initialization       : Warm start\n');
end
fprintf('Initial position SD  : %.2f m\n',   p_pos_0);
fprintf('Initial velocity SD  : %.2f m/s\n', p_vel_0);

fprintf('\n[ SPAWN CONFIGURATION ]\n');
if spawn_on
    switch measurement_type
        case 1, fprintf('Measurement type     : LiDAR\n');
        case 2, fprintf('Measurement type     : Range\n');
    end
    fprintf('SPAWN iterations     : %d\n', num_spawn_iter);

    if particles
        fprintf('Particle filtering   : Enabled\n');
        fprintf('Inter-sample particles: %d\n',   inter_samples);
        fprintf('Intra-sample particles: %d\n',   intra_samples);
        fprintf('Particle bandwidth h : %.2f\n',  particle_h);
        fprintf('Candidate constant kk: %d\n',    kk);
    else
        fprintf('Particle filtering   : Disabled\n');
    end
else
    fprintf('SPAWN                : Disabled\n');
end

fprintf('\n[ GPS CONFIGURATION ]\n');
if gps_activated
    fprintf('GPS                  : Enabled\n');
    fprintf('GPS interval         : %.2f s\n', gps_interval);
    fprintf('GPS standard deviation: %.2f m\n', gps_std);
else
    fprintf('GPS                  : Disabled\n');
end

fprintf('\n[ MEASUREMENT PARAMETERS ]\n');
fprintf('Velocity measurement SD: %.2f m/s\n', velocity_std);
if spawn_on && measurement_type == 1
    fprintf('LiDAR interval       : %.2f s\n', measurement_interval);
    fprintf('LiDAR detection range: %.2f m\n', lidar_d_max);
    fprintf('LiDAR measurement SD : %.2f m\n', lidar_std);
elseif spawn_on && measurement_type == 2
    fprintf('Range interval       : %.2f s\n', measurement_interval);
    fprintf('Range detection limit: %.2f m\n', range_d_max);
    fprintf('Range measurement SD : %.2f m\n', range_std);
end

fprintf('\n[ MEAN POSITION ERROR ]\n');
fprintf('                       Mean error (m)\n');
fprintf('                       --------------\n');
for i = 1:num_uavs
    fprintf('Drone %-3d             %10.4f\n', i, mean_error_per_drone(i));
end
fprintf('                       --------------\n');
fprintf('All drones             %10.4f m\n', mean_error_all);
if num_mc > 1
    fprintf('SD across MC runs      %10.4f m\n', std(mean_errors_mc));
end
fprintf('============================================================\n\n');

mean_error = mean_error_all;
