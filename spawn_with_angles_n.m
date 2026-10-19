load("true_acc_local.mat");
load("true_acc_global.mat");

load("x_true_state_global.mat");

load("true_theta.mat")
load("true_velocity_local.mat")

load("sim_data_sim1.mat");

sim_parameters = [1, 1, 20, 2];
lidar_activated = sim_parameters(1);
num_spawn_iter = sim_parameters(2);
lidar_range = sim_parameters(3);                     % detection range lidar (m)
lidar_interval = sim_parameters(4);
gps_activated = 1;
gps_interval = 10;

%% simulation parameters
T = sim_data(1);
dt_sim = sim_data(2);                            % simulation time step (s)
num_uavs = sim_data(3);

tvec_sim = 0:dt_sim:T;
steps_sim = length(tvec_sim);

dt = 0.5; %simulation step
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

sigma_theta = 0.005;

sigma_gps = 2;
gps_noise = sigma_gps^2 * eye(2);
gps_cov = gps_noise;

sigma_lidar = 0.2;                          % %measurement noise lidar
L_noise = sigma_lidar^2 * eye(2);
L_cov = L_noise;

sigma_vel =  0.2;                            % measurement noise velocity
V_noise = sigma_vel^2 * eye(2);
V_cov = V_noise;

sigma_process = 0.2;                        % motion model
Q_cov = sigma_process^2 * eye(4);

sigma_a = 0.2;                                % noise added to acceleration; 
A_noise = sigma_a^2 * eye(2);

P_0 = 2 * eye(4);                           % initial state covariance

%% initializations
z_lidar_true = cell(num_uavs, num_uavs);
lidar_points = cell(1, num_uavs);

x_est = cell(1, num_uavs);
for i = 1:num_uavs
    % initialize state estimate matrix with 0
    mu_pred{i} = zeros(4, steps);
    P_pred{i} = zeros(4, 4);
    %lidar_points{i} = zeros(2,1);

end

%% INI
num_mc = 1;
mean_errors_mc = zeros(num_mc,1);

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
    
        mu_pred{i}(:,1) = [x_true_state_global{i}(1:2,1); 0; 0] + [0.5*(i-1); 0.5*(i-1); 0; 0];
        P_pred{i} = P_0;
    
        traceP(i,1) = trace(P_pred{i});
    end

    for k = 2:1:(steps)
        index = floor((k-1)*dt / dt_sim) + 1;

        for i = 1:num_uavs
            theta_measured{i} = true_theta{i}(index) + sigma_theta * randn(1);

            P_old = P_pred{i};
            mu_old = mu_pred{i}(:,k-1);
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

            [P_new, mu_new, Rt_] = prediction_operation_with_angles(index, i, mu_old,P_old, theta_measured{i}, sigma_theta, vel_measured, u_measured, A, B, Q_cov, H_vel, V_cov, gps_measured, add_gps_measure);
            mu_new;
            P_new;
            mu_pred{i}(:,k) = mu_new;
            P_pred{i} = P_new;
               
            %store covariance
            traceP(i,k) = trace(P_pred{i});
        end
    
        %% SPAWN
    
        if rem(k,(lidar_interval/dt)) == 0 && lidar_activated == 1      %run only when lidar is active
            for i = 1:num_uavs
                % initial beliefs b0
                mu_local{i} = mu_pred{i}(:,k);
                P_local{i}  = P_pred{i};
            
                % SPAWN beliefs being broadcasted
                mu_spawn{i} = mu_local{i};
                p_spawn{i}  = P_local{i};
            
            end
            z_ij = cell(num_uavs, num_uavs);
            % retrieve lidar measurements 
            for i = 1:num_uavs
                for j = 1:num_uavs
                    if i == j
                        continue;
                    end
    
                    % if inside lidars range, generate measurement zij
                    if norm(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index)) < lidar_range 
                         real_theta = true_theta{i}(index);
                         rot = [cos(real_theta) -sin(real_theta); sin(real_theta) cos(real_theta)];
                         z_ij{i,j} = rot*(x_true_state_global{j}(1:2,index) - x_true_state_global{i}(1:2,index) + mvnrnd([0 0], L_noise)');
                    end
                end
            end
        
            for iter = 1:num_spawn_iter
    
                mu_spawn_new = cell(1,num_uavs);
                p_spawn_new  = cell(1,num_uavs);
    
                for i = 1:num_uavs
    
                    % canonical form of initial belief
                    lambda_i = inv(P_local{i});
                    eta_i = lambda_i * mu_local{i};

                    mu_i = mu_local{i};
                    P_i = P_local{i};
        
                    for j = 1:num_uavs
                        if i == j
                            continue;
                        end
    
                        if norm(x_true_state_global{j}(1:2, index) - x_true_state_global{i}(1:2, index)) < lidar_range
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
                    % store new beliefs to change when the iteration is over
                    p_spawn_new{i} = P_i;
                    mu_spawn_new{i} = mu_i;
            
                end
                % synchronous update of beliefs
                mu_spawn = mu_spawn_new;
                p_spawn  = p_spawn_new;
            
            end
            % store beliefs for next time step
            for i = 1:num_uavs
                mu_pred{i}(:,k) = mu_spawn{i};
                P_pred{i}       = p_spawn{i};
                traceP(i,k) = trace(P_pred{i});
          
            end
        end
    
        for i = 1:num_uavs
            pos_error{i}(:, k) = mu_pred{i}(1:2,k) - x_true_state_global{i}(1:2, index);
        end
    end
    
    error_total = 0;

    for i = 1:num_uavs
        for t = 1:k
            error_total = error_total + norm(pos_error{i}(:,t));
        end
    end

    mean_errors_mc(mc) = error_total / (num_uavs * k);
    
    % mean_error = error_total/(num_uavs*k)
end

mean_error = mean(mean_errors_mc)
%std_error  = std(mean_errors_mc)

disp(mean_error);

%% plot other thing
RGB = orderedcolors("gem"); 
color_array = {'r','m','b'};
figure; hold on; grid on;

for i = 1:num_uavs
    plot(x_true_state_global{i}(1,1:(steps_sim)), x_true_state_global{i}(2,1:(steps_sim)),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)
    plot(mu_pred{i}(1,1:(steps)), mu_pred{i}(2,1:(steps)), '-','Color', RGB(i,:), 'LineWidth',2)
    %plot(lidar_points{i}(1,:), lidar_points{i}(2,:), 'o','Color', RGB(i,:), 'LineWidth',2)
end

xlabel('x')
ylabel('y')
legend('Ref 1','Est 1','Ref 2','Est 2','Ref 3','Est 3')

figure
plot(tvec, traceP')
xlabel('time')
ylabel('trace(P)')
title('Covariance')
grid on
