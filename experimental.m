clc
clear

%% simulation parameters
num_uavs = 3;
num_spawn_iter = 0;

T = 30;                                % simulation duration (s)
dt = 0.1;                              % simulation time step (s)
tvec = 0:dt:T;
steps = length(tvec);

lidar_range = 30;                      % detection range lidar (m)
lidar_interval = 0.5;         

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

Hi = [-1 0 0 0;
     0 -1 0 0];

Hj = [1 0 0 0;
      0 1 0 0];

sigma_lidar = 0.2;                     % %measurement noise lidar
L_noise = sigma_lidar^2 * eye(2);
L_cov = L_noise;

sigma_vel = 0.5;                       % measurement noise velocity
V_noise = sigma_vel^2 * eye(2);
V_cov = V_noise;

sigma_process = 0.5;                   % motion model
Q_cov = sigma_process^2 * eye(4);

sigma_a = 1;                           % noise added to acceleration; 
A_noise = sigma_a^2 * eye(2);

P_0 = 2 * eye(4);                      % initial state covariance

%% initializations
z_lidar_true = cell(num_uavs, num_uavs);

wpts = cell(1,num_uavs);
tpts = cell(1,num_uavs);

pos_true = cell(1,num_uavs);
vel_true = cell(1,num_uavs);
acc_true = cell(1,num_uavs);

x_est = cell(1, num_uavs);
for i = 1:num_uavs
    x_est{i} = zeros(4, steps);        % initialize state estimate matrix with 0
end


%% generate trajectories
wpts{1,1} = [0  20 30;
    0  10  0];

wpts{1,2} = [15  0 5;
    10  5  -15];

wpts{1,3} = [-5  15 10;
    5  5  -5];

for i=1:num_uavs
    tpts{i} = [0 0.5*T T];
end

%% extract true positions, velocities and accelerations
for i = 1:num_uavs
    [pos_true{i}, vel_true{i}, acc_true{i}] = minjerkpolytraj(wpts{i}, tpts{i}, steps);

    x_true_state{i}(1:4,:) = [pos_true{i}; vel_true{i}];
    
    % uav 1 knows its starting position
    x_est{i}(:,1) = x_true_state{i}(1:4,1) + [0.5*(i-1);0.5*(i-1);0;0]; 
    
    mu_pred{i}(:,1) = x_est{i}(:,1);
    P_pred{i} = P_0;
end

%% SPAWN
for k = 2:steps
    for i = 1:num_uavs        
        P_old = P_pred{i};
        mu_old = mu_pred{i}(:,k-1);

        % add noise to acceleration
        u = acc_true{i}(:,k-1) + mvnrnd([0 0], A_noise).';  
        
        true_vel = x_true_state{i}(3:4,k);

        % get measurement of velocity with noise
        vel_measured = true_vel + mvnrnd([0 0], V_noise).';

        % prediction operation
        mu_old = A*mu_old + B*u;
        P_old  = A*P_old*A' + Q_cov;

        % motion model factor (converted to canonical form)
        lambda_m = inv(P_old);
        eta_m = lambda_m * mu_old;

        % likelihood factor
        lambda_l = H_vel' * (inv(V_cov)) * H_vel;
        eta_l = H_vel' * (inv(V_cov)) * vel_measured;

        % fusion of the factors (converted to moments form)
        lambda = lambda_m + lambda_l;
        eta = eta_m + eta_l;

        P_new = inv(lambda);
        mu_new = P_new * eta;

        mu_pred{i}(:,k) = mu_new;
        P_pred{i} = P_new;
           
        %store covariance
        traceP(i,k) = trace(P_pred{i});

    end

    %% SPAWN

    if rem(k,(lidar_interval/dt)) == 0 %run only when lidar is active
        for i = 1:num_uavs
            % initial beliefs b0
            mu_local{i} = mu_pred{i}(:,k);
            P_local{i}  = P_pred{i};
        
            % SPAWN beliefs being broadcasted
            mu_spawn{i} = mu_local{i};
            p_spawn{i}  = P_local{i};
        
        end
        
        % retrieve lidar measurements 
        for i = 1:num_uavs
            for j = 1:num_uavs
                if i == j
                    continue;
                end

                % if inside lidars range, generate measurement zij
                if norm(x_true_state{j}(1:2,k) - x_true_state{i}(1:2,k)) < lidar_range 
                        z_ij{i,j} = x_true_state{j}(1:2,k) - x_true_state{i}(1:2,k) + ...
                        mvnrnd([0 0], L_noise)';
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
    
                for j = 1:num_uavs
                    if i == j
                        continue;
                    end

                    if norm(x_true_state{j}(1:2, k) - x_true_state{i}(1:2, k)) < lidar_range
                       
                        % neighbors belief
                        mu_j = mu_spawn{j};
                        p_j = p_spawn{j};
        
                        % marginalize over xj
                        S_ij = Hj*p_j*Hj' + L_cov;
    
                        %z_eq = Hj*mu_j - z_ij{i,j};
                        z_eq = z_ij{i,j} - Hj*mu_j;
        
                        % convert to canonical form
                        lambda_meas = Hi' * inv(S_ij) * Hi;
                        eta_meas    = Hi' * inv(S_ij) * z_eq;
        
                        % fusion
                        lambda_i = lambda_i + lambda_meas;
                        eta_i = eta_i + eta_meas;
                    end
                end
                % store new beliefs to change when the iteration is over
                p_spawn_new{i} = inv(lambda_i);
                mu_spawn_new{i} = p_spawn_new{i} * eta_i;
        
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
            pos_error{i}(:, k) = abs(mu_pred{i}(1:2,k) - x_true_state{i}(1:2, k));   
    end
end

error_total = 0;
for i = 1:num_uavs
    for t = 1:k
        error_total = error_total + sqrt(pos_error{i}(1,t)^2 + pos_error{i}(2,t)^2);
    end
end

mean_error = error_total/(num_uavs*k)

%% plot other thing
RGB = orderedcolors("gem"); 
color_array = {'r','m','b'};
figure; hold on; grid on;

for i = 1:num_uavs
    plot(x_true_state{i}(1,:), x_true_state{i}(2,:),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)
    plot(mu_pred{i}(1,:), mu_pred{i}(2,:), 'Color', RGB(i,:), 'LineWidth',2)
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
