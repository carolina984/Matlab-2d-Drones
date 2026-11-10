clc
clear
T = 120;                              % simulation duration (s)
dt_sim = 0.01;                              % simulation time step (s)
tvec_sim = 0:dt_sim:T;
steps_sim = length(tvec_sim);
num_uavs = 3;

sim_data = [T; dt_sim; num_uavs];
num_uavs = 3;
wpts{1,1} = [50  0 -40 0 100;
            70  70  30 -15 -15];

wpts{1,2} = [50  0 -45 0 100;
            75  75  30 -20 -20];

wpts{1,3} = [50  0 -35 0 100;
            65  65  30 -10 -10];

theta_lim(1, 1:2) = [pi/2, 3*pi/2];
theta_lim(2, 1:2) = [0 , pi/2];
theta_lim(3, 1:2) = [0, pi];

for i=1:num_uavs
    tpts{i} = [0 0.3*T 0.5*T 0.7*T T];
end

%% extract true positions, velocities and accelerations
for i = 1:num_uavs
    [pos_true{i}, vel_true{i}, acc_true{i}] = minjerkpolytraj(wpts{i}, tpts{i}, steps_sim);

    x_true_state_global{i}(1:5,:) = [pos_true{i}; vel_true{i}; tvec_sim];
    true_acc_global{i}(1:3,:) = [acc_true{i};tvec_sim];
    true_theta{i}(:) = theta_lim(i, 1): (theta_lim(i, 2)-theta_lim(i, 1))/(steps_sim-1) :theta_lim(i, 2);

    for k = 1:1:steps_sim
        theta = true_theta{i}(k)
        rotation_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        rotation_matrix*vel_true{i}(1:2, k);
        true_velocity_local{i}(1:3,k) = [rotation_matrix*vel_true{i}(1:2, k); tvec_sim(k)];
        true_acc_local{i}(1:3,k) = [rotation_matrix*acc_true{i}(1:2, k);tvec_sim(k)];
    end

end

% for k = 0:1:steps_sim


RGB = orderedcolors("gem"); 
color_array = {'r','m','b'};
figure; hold on; grid on;

for i = 1:num_uavs
    plot(x_true_state_global{i}(1,1:(steps_sim)), x_true_state_global{i}(2,1:(steps_sim)),'Color', RGB(i,:), 'LineStyle','--', 'LineWidth',2)
end

xlabel('x')
ylabel('y')
legend('Ref 1', 'Ref 2', 'Ref 3')

