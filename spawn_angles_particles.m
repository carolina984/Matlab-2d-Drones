clear

load("true_acc_local.mat");
load("true_acc_global.mat");

load("x_true_state_global.mat");

load("true_theta.mat")
load("true_velocity_local.mat")

load("sim_data_sim1.mat");


a = 0;
b = 0;


kk = 2;                                                                       %sampling constant


%% Simulation Selection 

% 0 - none, 1 - lidar, 2 - range
% number of spawn iterations 
% particles? 0 - no, 1 - yes
% gps on? 0 - no, 1 - yes
% warm start - 0, cold start - 1

spawn_parameters = [2, 3, 1, 1, 1];

% particles
inter_samples = 250;
intra_samples = 1000;
particle_h = 0.3;

% measurement noise velocity
velocity_std = 0.4; 

% measurements
measurement_interval = 2;

% range
range_d_max = 30;
range_std = 0.3;

% lidar
lidar_d_max = 30;               % detection range lidar (m)
lidar_std = 0.1;

% gps
gps_interval = 5;
gps_std = 2;


%% Extract Data - Aux
measurement_type = spawn_parameters(1);
spawn_on = measurement_type ~= 0;

if spawn_on == true
    num_spawn_iter = spawn_parameters(2);
    if measurement_type == 2
        particles = spawn_parameters(3);
    else
        particles = 0;
    end
else
    num_spawn_iter = 0;
    measurement_type = 0;
end

gps_activated = spawn_parameters(4);

cold_start = boolean(spawn_parameters(5));

%% Simulation parameters

T = sim_data(1);
dt_sim = sim_data(2);           % simulation time step (s)
num_uavs = sim_data(3);
number_agents = num_uavs;
number_anchors = 0;

tvec_sim = 0:dt_sim:T;
steps_sim = length(tvec_sim);

dt = 0.5;                       % simulation step
tvec = 0:dt:T;
steps = length(tvec);

A = [1 0 dt 0;
    0 1 0 dt;
    0 0 1 0;
    0 0 0 1];

B = [0.5*dt^2 0;
    0 0.5*dt^2;
    dt 0;
    0 dt];

H_gps = [1 0 0 0;
        0 1 0 0];

H_vel = [0 0 1 0;
        0 0 0 1];

Hi = [-1 0 0 0;
     0 -1 0 0];

Hj = [1 0 0 0;
      0 1 0 0];

sigma_theta = 0.05;

gps_noise = gps_std^2 * eye(2);
gps_cov = gps_noise;
                       
R_noise = range_std;

L_noise = lidar_std^2 * eye(2); % measurement noise lidar
L_cov = L_noise;

V_noise = velocity_std^2 * eye(2);
V_cov = V_noise;

sigma_process = 0.3;            % motion model
Q_cov = sigma_process^2 * eye(4);

sigma_a = 0.3;                  % noise added to acceleration; 
A_noise = sigma_a^2 * eye(2);

if cold_start
    p_pos_0 = 10.0;             % initial position sd [m]
    p_vel_0 = 1.0;              % initial velocity sd [m/s]
else
    p_pos_0 = 1.0;              % initial position sd [m]
    p_vel_0 = 1.0;              % initial velocity sd [m/s]
end

P_0 = diag([p_pos_0^2, p_pos_0^2, p_vel_0^2, p_vel_0^2]);

%% Initializations - Aux
z_lidar_true = cell(num_uavs, num_uavs);
lidar_points = cell(1, num_uavs);

x_est = cell(1, num_uavs);

%% Initial beliefs
num_mc = 20;

mean_errors_mc = zeros(num_mc, 1);
mean_errors_per_drone_mc = zeros(num_mc, num_uavs);

for mc = 1:num_mc
    rng(mc);
    mu_pred = cell(1, num_uavs);
    theta_measured = cell(1, num_uavs);
    P_pred  = cell(1, num_uavs);
    traceP  = zeros(num_uavs, steps);
    pos_error = cell(1, num_uavs);
    mu_local = cell(1,num_uavs);
    P_local  = cell(1,num_uavs);
    mu_spawn = cell(1,num_uavs);
    p_spawn  = cell(1,num_uavs);
    lidar_points = cell(1, num_uavs);

    for i = 1:num_uavs
        mu_pred{i} = zeros(4, steps);
        P_pred{i}  = zeros(4, 4);

        e_0 = mvnrnd(zeros(4,1), P_0)';

        if particles
            bel_agents{i} = gaussian_belief(x_true_state_global{i}(1:4,1) + e_0, P_0, intra_samples);
            
            [mu0, P0_est] = belief_estimate(bel_agents{i});
            mu_pred{i}(:,1) = mu0(:);
            P_pred{i}       = P0_est;
        else

            mu_pred{i} = zeros(4, steps);
            P_pred{i}  = zeros(4, 4);

            mu_pred{i}(:,1) = x_true_state_global{i}(1:4,1) + e_0;
            P_pred{i} = P_0;
        end

        traceP(i,1) = trace(P_pred{i});
        pos_error{i}(:,1) = mu_pred{i}(1:2,1) - x_true_state_global{i}(1:2,1);
    end

    for k = 2:1:(steps)
        index = floor((k-1)*dt / dt_sim) + 1;

        % Run initial Kalman like Update
        for i = 1:num_uavs
            theta_measured{i} = true_theta{i}(index) + sigma_theta * randn(1);

            % P_old = P_pred{i};
            % mu_old = mu_pred{i}(:,k-1);
            
            R__ = [cos(theta_measured{i}) sin(theta_measured{i}); -sin(theta_measured{i}) cos(theta_measured{i})];
            % add noise to acceleration
            u_measured = true_acc_local{i}(1:2,index) + mvnrnd([0 0], A_noise).';
    
            % get measurement of velocity with noise
            vel_measured = true_velocity_local{i}(1:2,index) + mvnrnd([0 0], V_noise).';
            
            error_outside = (R__*true_acc_local{i}(1:2,index) - true_acc_global{i}(1:2,index));
            true_pos = x_true_state_global{i}(1:2,index);
            
            % get measurement of velocity with noise
            gps_measured = true_pos + mvnrnd([0 0], gps_noise).';
            

            if rem(k,(gps_interval/dt)) == 0 && gps_activated == 1 && i == 2
                add_gps_measure = 1;
                
            else
                add_gps_measure = 0;
            end

            if particles
                belief_old = bel_agents{i};
                [bel_agents{i}, ess_pred(i,k)] = prediction_operation_particles(belief_old, theta_measured{i}, sigma_theta, vel_measured, u_measured, A, B, Q_cov, V_cov, gps_measured, add_gps_measure, gps_cov);
                [mu_pred{i}(:,k), P_pred{i}] = belief_estimate(bel_agents{i});
                broadcasted_message{i} = intrasamples_to_intersamples(bel_agents{i}, inter_samples);
            else
                P_old = P_pred{i};
                mu_old = mu_pred{i}(:,k-1);
                [P_new, mu_new, Rt_] = prediction_operation_with_angles(index, i, mu_old,P_old, theta_measured{i}, sigma_theta, vel_measured, u_measured, A, B, Q_cov, H_vel, V_cov, gps_measured, add_gps_measure);
                mu_new;
                P_new;
                mu_pred{i}(:,k) = mu_new;
                P_pred{i} = P_new;
            
            end
            
            
            % mu_new;
            % P_new;
            % mu_pred{i}(:,k) = mu_new;
            % P_pred{i} = P_new;
               
            %store covariance
            traceP(i,k) = trace(P_pred{i});
        end

        % SPAWN UPDATE
        if spawn_on && rem(k, round(measurement_interval/dt)) == 0
            % Initialize local beliefs and broadcast beliefs
            for i = 1:num_uavs
                mu_local{i} = mu_pred{i}(:,k);
                P_local{i}  = P_pred{i};
        
                mu_spawn{i} = mu_local{i};
                p_spawn{i}  = P_local{i};
            end
        
            % Choose algorithm based on range/ lidar and particles ot not
            switch measurement_type
       
                case 1 % CASE 1: LiDAR SPAWN (NO PARTICLES)
                    % Generate LiDAR measurements between UAVs
                    z_ij = cell(num_uavs, num_uavs);
        
                    for i = 1:num_uavs
                        for j = 1:num_uavs
                            if i == j
                                continue;
                            end
    
                            % if inside lidars range, generate measurement zij
                            if norm(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index)) < lidar_d_max 
                                 real_theta = true_theta{i}(index);
                                 rot = [cos(real_theta) -sin(real_theta); sin(real_theta) cos(real_theta)];
                                 z_ij{i,j} = rot*(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index) + mvnrnd([0 0], L_noise)');
                            end
                        end
                    end
        
                    % SPAWN iterations
                    for iter = 1:num_spawn_iter
        
                        mu_spawn_new = cell(1,num_uavs);
                        p_spawn_new  = cell(1,num_uavs);
        
                        for i = 1:num_uavs
        
                            mu_i = mu_local{i};
                            P_i  = P_local{i};
        
                            for j = 1:num_uavs
                                if i == j
                                    continue;
                                end
        
                                if norm(x_true_state_global{j}(1:2, index) - x_true_state_global{i}(1:2, index)) < lidar_d_max
                                    %lidar_points{i}(:,k) = mu_pred{i}(1:2, k-1);
                                    lidar_points{i} = [lidar_points{i}(:,:),mu_pred{i}(1:2, k-1)];
                                    % neighbors belief
                                    mu_j = mu_spawn{j};
                                    p_j = p_spawn{j};
        
        
                                    [mu_i, P_i] = intermeasurements_lidar(theta_measured{i}, sigma_theta, mu_j, p_j, mu_i, P_i, L_cov, z_ij{i,j});
                                    mu_i;
                                    P_i;
                                end
                            end
        
                            % Store beliefs for synchronous updating
                            mu_spawn_new{i} = mu_i;
                            p_spawn_new{i}  = P_i;
                        end
        
                        % Synchronous update
                        mu_spawn = mu_spawn_new;
                        p_spawn  = p_spawn_new;
                    end
              
                case 2 % CASE 2: RANGE SPAWN   
                    % Generate range measurements between UAVs
                    z_range = cell(num_uavs, num_uavs);
 
                    for i = 1:num_uavs
                        for j = 1:num_uavs
                            if i == j
                                continue;
                            end
    
                            % if inside range, generate measurement zij
                            if norm(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index)) < range_d_max
                                 z_range{i,j} = norm(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index)) + normrnd(0, range_std); 
                            end
                        end
                    end

                    if particles
                        % SPAWN iterations - Particles
                        pred_bel_agents = bel_agents;          % freeze the prior, like mu_local
                      
                        for f = 1:num_spawn_iter
                            a = 0;
                            b = 0;
                            old_bel_agents = bel_agents; 
                            old_broadcast = broadcasted_message;        % frozen for this iteration
                            new_broadcast = cell(1, number_agents);
                            for i = 1:1:number_agents                                                %for each agente, in this case only one
                                a = 0;
                                b = 0;
                                message_agent  = {};                                                 % add these two lines
                                message_anchor = {};
                           
                                % for j = 1:1:number_anchors
                                %     if i == j
                                %         continue;
                                %     end
                                % 
                                %     range_to_anchors(i, j)
                                %     b = b + 1;
                                %     message_anchor{b} = compute_message(bel_anchors{j}, inter_samples, intra_samples, z_range{i,j}, 0.1);
                                % 
                                % end
                                
                                if f>0
                                    for j = 1:number_agents
                                        if i == j
                                            continue;
                                        end
                                        if ~isempty(z_range{i,j}) && old_bel_agents{j}(1,6) ~= 0 %its a neighbor! and they ahve information 
                                            a = a + 1;
                                            message_agent{a} = compute_message(old_broadcast{j}, inter_samples, intra_samples, z_range{i,j}, range_std);
                                        end
                                    end
                                    message_anchor = [message_anchor, message_agent];
                                end
                                [new_result, new_message] = product_messages(f, pred_bel_agents{i}, message_anchor, intra_samples, inter_samples, kk, intra_samples, particle_h);
                                new_bel_agents{i} = new_result;
                                new_broadcast{i} = new_message;
                            end
                            bel_agents = new_bel_agents;
                            broadcasted_message = new_broadcast;

                            % Synchronous update
                            for i=1:num_uavs
                                [mu_spawn{i}, p_spawn{i}, sigma__] = belief_estimate(bel_agents{i});
                            end

                        end

                    else
                        % SPAWN iterations - No particles
                        for iter = 1:num_spawn_iter
            
                            mu_spawn_new = cell(1,num_uavs);
                            p_spawn_new  = cell(1,num_uavs);
            
                            for i = 1:num_uavs
            
                                % Start with the local belief
                                mu_i = mu_local{i};
                                P_i  = P_local{i};
    
                                for j = 1:num_uavs                               
            
                                    if i == j
                                        continue;
                                    end
    
                                    if norm(x_true_state_global{j}(1:2, index) - x_true_state_global{i}(1:2, index)) < range_d_max
                                        mu_j = mu_spawn{j};
                                        p_j  = p_spawn{j};
                                        
                                        if particles == false
                                            [mu_i, P_i] = intermeasurement_range_linearized(mu_j, p_j, mu_i, P_i, range_std, z_range{i,j});
                                        % else 
                                        %     [mu_i, P_i] = intermeasurement_range_particles(theta_measured{i}, sigma_theta, mu_j, p_j, mu_i, P_i, L_cov, z_ij{i,j});
                                        end
                                        mu_i;
                                        P_i;
                                    end
                                    
                                    % Keep the two methods separate.
                                end
            
                                mu_spawn_new{i} = mu_i;
                                p_spawn_new{i}  = P_i;
                            end
            
                            % Synchronous update
                            mu_spawn = mu_spawn_new;
                            p_spawn  = p_spawn_new;
                        end
                    end
        
            end
          
            % Store SPAWN beliefs
            for i = 1:num_uavs
                mu_pred{i}(:,k) = mu_spawn{i};
                P_pred{i}       = p_spawn{i};
        
                traceP(i,k) = trace(P_pred{i});
            end
        end
    
        % Store error in a given timestep
        for i = 1:num_uavs
            if particles
                [mu__, P__, sigma__] = belief_estimate(bel_agents{i});
                pos_error{i}(:, k) = mu__(1:2).' - x_true_state_global{i}(1:2, index);
            else
                pos_error{i}(:, k) = mu_pred{i}(1:2,k) - x_true_state_global{i}(1:2, index);
            end
            
        end

    end

    % Mean position error for each drone in this Monte Carlo run
    for i = 1:num_uavs
        errors_i = vecnorm(pos_error{i}(:,1:k), 2, 1);
        mean_errors_per_drone_mc(mc,i) = mean(errors_i);
    end

    % Overall mean position error for this Monte Carlo run
    mean_errors_mc(mc) = mean(mean_errors_per_drone_mc(mc,:));
end


%% plot other thing
RGB = orderedcolors("gem"); 
color_array = {'r','m','b'};
figure; hold on; grid on;

for i = 1:num_uavs
    plot(x_true_state_global{i}(1,1:(steps_sim)), x_true_state_global{i}(2,1:(steps_sim)),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)
    plot(mu_pred{i}(1,1:(steps)), mu_pred{i}(2,1:(steps)), '-','Color', RGB(i,:), 'LineWidth',2)
    %plot(lidar_points{i}(1,:), lidar_points{i}(2,:), 'o','Color', RGB(i,:), 'LineWidth',2)
end

% for i = 1:1:number_agents                                                    %for each agente, in this case only one
%     scatter(bel_agents{i}(:,1), bel_agents{i}(:,2), 20, '*')
%     hold on;
% end

xlabel('x')
ylabel('y')
legend('Ref 1','Est 1','Ref 2','Est 2','Ref 3','Est 3')

figure
plot(tvec, traceP')
xlabel('time')
ylabel('trace(P)')
title('Covariance')
grid on


%% SIMULATION REPORT

% Aggregate results across Monte Carlo runs
mean_error_per_drone = mean(mean_errors_per_drone_mc, 1);
mean_error_all = mean(mean_errors_mc);

% Report header
fprintf('\n');
fprintf('============================================================\n');
fprintf('                    SIMULATION REPORT\n');
fprintf('============================================================\n');
fprintf('Run date: %s\n', datetime("now"));
fprintf('Monte Carlo simulations: %d\n', num_mc);
fprintf('------------------------------------------------------------\n');

% Initialization
if cold_start
    fprintf('Initialization       : Cold start\n');
else
    fprintf('Initialization       : Warm start\n');
end
fprintf('Initial position SD  : %.2f m\n', p_pos_0);
fprintf('Initial velocity SD  : %.2f m/s\n', p_vel_0);

% SPAWN configuration
fprintf('\n[ SPAWN CONFIGURATION ]\n');

if spawn_on
    switch measurement_type
        case 1
            fprintf('Measurement type     : LiDAR\n');
        case 2
            fprintf('Measurement type     : Range\n');
    end

    fprintf('SPAWN iterations     : %d\n', num_spawn_iter);

    if particles
        fprintf('Particle filtering   : Enabled\n');
        fprintf('Inter-sample particles: %d\n', inter_samples);
        fprintf('Intra-sample particles: %d\n', intra_samples);
        fprintf('Particle bandwidth h : %.2f\n', particle_h);
    else
        fprintf('Particle filtering   : Disabled\n');
    end
else
    fprintf('SPAWN                 : Disabled\n');
end

% GPS configuration
fprintf('\n[ GPS CONFIGURATION ]\n');

if gps_activated
    fprintf('GPS                  : Enabled\n');
    fprintf('GPS interval         : %.2f s\n', gps_interval);
    fprintf('GPS standard deviation: %.2f m\n', gps_std);
else
    fprintf('GPS                  : Disabled\n');
end

% Measurement configuration
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

% Error results
fprintf('\n[ MEAN POSITION ERROR ]\n');
fprintf('                       Mean error (m)\n');
fprintf('                       --------------\n');

for i = 1:num_uavs
    fprintf('Drone %-3d             %10.4f\n', ...
        i, mean_error_per_drone(i));
end

fprintf('                       --------------\n');
fprintf('All drones             %10.4f m\n', mean_error_all);

if num_mc > 1
    fprintf('SD across MC runs      %10.4f m\n', std(mean_errors_mc));
end

fprintf('============================================================\n\n');

% Keep the overall mean available for subsequent code
mean_error = mean_error_all;
