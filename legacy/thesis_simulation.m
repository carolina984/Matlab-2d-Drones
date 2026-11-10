clc
clear

%% SIMULATION OPTIONS
opt = 2;

% option 0 -> just motion model
% option 1 -> add zself (velocimeter)
% option 2 -> add zij (lidar)

%% simulation parameters
num_uavs = 3;
num_spawn_iter = 2;

T = 20;                         % simulation duration (s)
dt = 0.05;                      % simulation time step (S)
tvec = 0:dt:T;
steps = length(tvec);

lidar_range = 10;               % detection range LIDAR (m)

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

H_ij = [1 0 0 0;
       0 1 0 0];

Hi = [-1 0 0 0;
      0 -1 0 0];

Hj = [1 0 0 0;
      0 1 0 0];

Rr = 0.01*eye(2);

sigma_process = 2;
sigma_vel = 1;
Q = sigma_process^2 * [dt^4/4 0 dt^3/2 0; 0 dt^4/4 0 dt^3/2; dt^3/2 0 dt^2 0; 0 dt^3/2 0 dt^2];

%% initializations
z_lidar_true = cell(num_uavs, num_uavs);

wpts = cell(1,num_uavs);
tpts = cell(1,num_uavs);

pos_true = cell(1,num_uavs);
vel_true = cell(1,num_uavs);
acc_true = cell(1,num_uavs);

x_est = cell(1, num_uavs);
for i = 1:num_uavs
    x_est{i} = zeros(4, steps); %initialize state estimate matrix with 0
    for j = 1:num_uavs
        z_lidar_true{i, j} = NaN(2,steps);
    end
end


%% trajectories
wpts{1,1} = [0  10 20;
    0  10  0];

wpts{1,2} = [10  0 10;
    10  5  5];

wpts{1,3} = [5  15 10;
    5  5  0];

for i=1:num_uavs
    tpts{i} = [0 0.5*T T];
end

%% obtain true positions, velocities and accelerations for the given traject
for i = 1:num_uavs
    [pos_true{i}, vel_true{i}, acc_true{i}] = minjerkpolytraj(wpts{i}, tpts{i}, steps);
    x_true_state{i}(1:4,:) = [pos_true{i}; vel_true{i}];
end

%% initial state estimate
for i = 1:num_uavs
    %x_est{i}(:,1) = x_true_state{i}(1:4, 1);
    x_est{i}(:,1) = x_true_state{i}(1:4,1) + [3;3;1;1].*randn(4,1);
    mu_pred{i}(:,1) = x_est{i}(:,1);
    P_pred{i} = 0.1 * eye(4);
    mu_pred_spawn{i}(:,1) = mu_pred{i}(:,1);
    P_pred_spawn{i} = P_pred{i};
end

%% simulation
for k = 2:steps
    for i = 1:num_uavs

        sigma_a = 0.1;          % m/s^2
        W = sigma_a^2 * eye(2);
        w = mvnrnd([0 0], W).';
        u = acc_true{i}(:,k-1) + w;

        %% =========================
        %% TRUE DYNAMICS (NOW NOISY)
        %% =========================
        true_vel = x_true_state{i}(3:4,k);

        %% sensor measurement (velocity)
        R_vel = sigma_vel^2 * eye(2);
        vel_measured = true_vel + mvnrnd([0 0], R_vel).';

        %% =========================
        %% SPAWN FILTER
        %% =========================

        P_old = P_pred{i};
        mu_old = mu_pred{i}(:,k-1);

        % PREDICTION
        mu_old = A*mu_old + B*u;
        P_old  = A*P_old*A' + Q;

        if opt == 0
            mu_new = mu_old;
            P_new  = P_old;
        else 
            % motion factor (information form)
            Lambda_m = inv(P_old);
            eta_m = Lambda_m * mu_old;
    
            % likelihood factor
            R = R_vel;
    
            Lambda_l = H_vel' * (inv(R)) * H_vel;
            eta_l = H_vel' * (inv(R)) * vel_measured;
    
            % SPAWN fusion
            Lambda = Lambda_m + Lambda_l;
            eta = eta_m + eta_l;
    
            P_new = inv(Lambda);
            mu_new = P_new * eta;
        end

        mu_pred{i}(:,k) = mu_new;
        P_pred{i} = P_new;
           

        %CORRECTION
        %belief
        traceP(i,k) = trace(P_new);

    end

    %SPAWN
    if opt == 2
        for i = 1:num_uavs
            mu_local{i} = mu_pred{i}(:,k);
            P_local{i} = P_pred{i};
        
            mu_spawn{i} = mu_local{i};
            p_spawn{i} = P_local{i};
            for j = 1:num_uavs
                z_ij{i, j} = x_true_state{j}(1:2, k) - x_true_state{i}(1:2, k) + mvnrnd([0 0],Rr)';
            end

        end

        for iter = 1:num_spawn_iter

                for i = 1:num_uavs
                    lambda_i = inv(p_spawn{i});
                    eta_i = lambda_i * mu_spawn{i};

                    for j = 1:num_uavs 
                        if i == j
                            continue;
                        end

                        % relative measurement
                        if norm(x_true_state{j}(1:2, k) - x_true_state{i}(1:2, k)) < lidar_range
                           
                            %% neighbor belief
                            mu_j = mu_spawn{j};
                            p_j = p_spawn{j}
            
                            %% marginalize neighbor
    
                            %p_i = P_local{i};

                            S_ij = Hj*p_j*Hj' + Rr;
            
                            % equivalent measurement model
            
                            H_eq = Hi;
            
                            z_eq = z_ij{i, j} - Hj*mu_j;
            
                            %% information contribution
            
                            Lambda_meas = H_eq' / S_ij * H_eq;
            
                            eta_meas = H_eq' / S_ij * z_eq;
            
                            %% accumulate
            
                            lambda_i = lambda_i + Lambda_meas;
            
                            eta_i = eta_i + eta_meas;
                        end
                    end
                    p_spawn{i} = inv(lambda_i);
                    mu_spawn{i} = p_spawn{i} * eta_i;
                end
        end
        for i = 1:num_uavs
            mu_pred_spawn{i}(:,k) = mu_spawn{i};
            P_pred_spawn{i} = p_spawn{i};
            
        end
        % 
        % mu_pred_spawn{i}(:,k) = mu_spawn{i};
        % P_pred_spawn{i} = P_spawn{i};
    end

end


%% plot other thing
RGB = orderedcolors("gem");     % optional nicer colors
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
title('Covariance growth')
grid on

figure; hold on; grid on;

for i = 1:num_uavs

    plot(x_true_state{i}(1,:), x_true_state{i}(2,:),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)

    plot(mu_pred_spawn{i}(1,:), mu_pred_spawn{i}(2,:), 'Color', RGB(i,:), 'LineWidth',2)
end

xlabel('x')
ylabel('y')
legend('Ref 1','Est 1','Ref 2','Est 2','Ref 3','Est 3')

figure
plot(tvec, traceP')
xlabel('time')
ylabel('trace(P)')
title('Covariance growth')
grid on
