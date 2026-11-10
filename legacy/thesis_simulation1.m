clc
clear

%% simulation parameters
num_uavs = 3;

T = 20;                                 % simulation duration (s)
dt = 0.05;                              % simulation time step (S)
tvec = 0:dt:T;
steps = length(tvec);

max_range = 10;                         % detection range LIDAR (m)

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

sigma_process = 0.05;
sigma_vel = 0.002;
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
    x_est{i} = zeros(4, steps);         %initialize state estimate matrix with 0
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
    x_true{i} = [pos_true{i}; vel_true{i}];
end

%% initial state estimate
for i = 1:num_uavs
    x_est{i}(:, 1) = x_true{i}(:, 1);   %get first estimate
    mu_pred{i}(:, 1) = x_true{i}(:, 1); %get first estimate
    P_pred{i} = 0.1 * eye(4);
end

%% simulation
for k = 2:steps
    for i = 1:num_uavs
        u = acc_true{i}(:, k-1);        % control input (ax, ay)
        % P_prev = P_pred{i};
        % 
        % 
        % traceP(i,k) = trace(P_pred{i});
        % 
        % 
        % 
        % mu_old = A*(mu_pred{i}(:, k-1)) + B*u
        % P_old = A * P_prev * A' + Q
        % 
        % 
        % Lambda_m = inv(P_old);
        % eta_m = Lambda_m * mu_old;
        % 
        % %measurements
        % % mu_pred{i}(1:2, k)
        % % mu_pred{i}(1:2, k-1)
        w = mvnrnd([0 0 0 0], Q).';
        x_true{i}(:,k) = A*x_true{i}(:,k-1) + B*u + w;
        true_vel = x_true{i}(3:4, k);
        R_vel = sigma_vel^2 * eye(2);
        vel_measured = true_vel + mvnrnd([0 0], R_vel).';
        % 
        % R = 1*R_vel;
        % Lambda_l = H_vel' * (1/R) * H_vel * 0.5;
        % eta_l = H_vel' * inv(R) * vel_measured;
        % % h = (1/dt) * [mu_pred{i}(1:2, k) - mu_pred{i}(1:2, k-1)]
        % % z_vel = h + mvnrnd([0 0], R_vel)'
        % 
        % Z = 1*R_vel;
        % % mu_new = H_vel* mu_old;
        % % p_new = H_vel * P_old * H_vel' + Z
        % 
        % 
        % Lambda = Lambda_m + Lambda_l;
        % eta = eta_m + eta_l;
        % 
        % P_pred{i} = inv(Lambda);
        % mu_new = P_new * eta
        % %z_self = z_vel;      
        % 
        % mu_pred{i}(:, k) = mu_new;
        % 
        % % mu_pred{i}(:, k) = [mu_old(1:2, 1); mu_new];
        % % [P_old(1:2, 1:4); P_old(3, 1:2) p_new(1, 1:2); P_old(4, 1:2) p_new(2, 1:2)]
        % % P_pred{i} = [P_old(1:2, 1:4); P_old(3, 1:2) p_new(1, 1:2); P_old(4, 1:2) p_new(2, 1:2)];
        traceP(i,k) = trace(P_pred{i});
        P_old = P_pred{i};
        mu_old = mu_pred{i}(:, k-1);
        
        % prediction
        mu_old = A*mu_old + B*u;
        P_old  = A*P_old*A' + Q;
        
        Lambda_m = inv(P_old);
        eta_m = Lambda_m * mu_old;
        
        % measurement
        z = vel_measured;
        R = sigma_vel^2 * eye(2);
        
        Lambda_l = H_vel' * inv(R) * H_vel;
        eta_l = H_vel' * inv(R) * z;
        
        % SPAWN fusion
        Lambda = Lambda_m + Lambda_l;
        eta = eta_m + eta_l;
        
        P_new = inv(Lambda);
        mu_new = P_new * eta;
        
        mu_pred{i}(:, k) = mu_new;
        P_pred{i} = P_new;

    end
end

%% animate uavs trajectories
% fig = figure;
% hold on;
% grid on;
% axis equal;
% colors = ['r','g','b'];
%
% ylim([0 300]);
% axis equal;
% axis manual;
%
% for t = 1:(0.2/dt):steps
%     if ~isvalid(fig)
%         break;
%     end
%
%     e = 1;
%
%     for i = 1:num_uavs
%         true_pos_self = [squeeze(uavs_true_positions(t,i,1)), squeeze(uavs_true_positions(t,i,2))];
%
%         for a = i+1:num_uavs
%             if (a ~= i)
%                 true_pos_neighbor = [squeeze(uavs_true_positions(t,a,1)), squeeze(uavs_true_positions(t,a,2))];
%                 dist = norm(true_pos_neighbor - true_pos_self);
%                 if (dist > max_range)
%                     range_matrix(i, a) = NaN;
%                     range_matrix(a, i) = NaN;
%                 else
%                     edge(e) = line([true_pos_self(1) true_pos_neighbor(1)], [true_pos_self(2) true_pos_neighbor(2)]);
%                     edge(e).Color = 'green';
%                     e = e +1;
%                     range_matrix(i, a) = dist;
%                     range_matrix(a, i) = range_matrix(i, a);
%                 end
%
%             end
%
%         end
%
%         h(i) = plot(true_pos_self(1), true_pos_self(2), [colors(i) 'o'], 'LineWidth', 2);
%         j(i) = viscircles(true_pos_self,max_range,'Color',colors(i));
%     end
%
%     pause(0.01)
%     delete(h)
%     delete(edge)
%     delete(j)
% end
% hold off
%
% if isvalid(fig)
%     close(fig)
% end
%

%% plot other thing
RGB = orderedcolors("gem");             % optional nicer colors
color_array = {'r','m','b'};

figure; hold on; grid on;

for i = 1:num_uavs

    plot(pos_true{i}(1,:), pos_true{i}(2,:),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)

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
