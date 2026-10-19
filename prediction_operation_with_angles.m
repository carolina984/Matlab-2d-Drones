function [P_new, mu_new, Rt_] = prediction_operation_with_angles(index, i, mu_old,P_old, theta, sigma_theta, vel_measured, u, A, B, Q, H_vel, V_cov, gps_measured, add_gps)
    sigma_gps = 2;
    gps_noise = sigma_gps^2 * eye(2);
    gps_cov = gps_noise;
    H_gps = [1 0 0 0;
        0 1 0 0];


    R_ = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    Rt_ = [cos(theta) sin(theta); -sin(theta) cos(theta)];
    dR_ = [0 -1; 1 0] * R_;
    dRt_ = [-sin(theta) cos(theta); -cos(theta) -sin(theta)];

    % prediction operation
    mu_old = A*mu_old + B*(Rt_*u);
    %P_old  = A*P_old*A' + Q_cov;
    %sigma_a = 0.4;                                % noise added to acceleration; 
    %A_cov = sigma_a^2 * eye(2);
    sigma_a = 0.1;                                % noise added to acceleration; 
    A_noise = sigma_a^2 * eye(2);

    sigma_process = 0.2;                        % motion model
    Q_cov = sigma_process^2 * eye(4);

    P_old  = A*P_old*A' + (B*dRt_*u)*sigma_theta^2*(u'*dR_*B') + Q_cov; % (B*dRt_*A_noise'*dR_*B'); 
    %P_0 = 1 * eye(4);
    % motion model factor (converted to canonical form)
    lambda_m = inv(P_old);
    eta_m = lambda_m * mu_old;

    % likelihood factor VELOCITY

    C = (dR_ * H_vel * mu_old * sigma_theta^2 * mu_old' * H_vel' * dRt_) + R_ * V_cov* Rt_;

    % lambda_v = H_vel' * (inv(V_cov)) * H_vel;
    com = H_vel' * Rt_ * (inv(C)) * vel_measured - H_vel' * (inv(V_cov)) * Rt_ * vel_measured;
    dif_lamda = H_vel' * (inv(V_cov)) * H_vel - H_vel' * Rt_*(inv(C)) * R_ *H_vel;
    

    % lambda_v = H_vel' * (inv(V_cov)) * H_vel;
    % eta_v = H_vel' * (inv(V_cov)) * Rt_ * vel_measured;
    
    lambda_v = H_vel' * Rt_*(inv(C)) * R_ *H_vel;
    eta_v = H_vel' * Rt_ * (inv(C)) * vel_measured;

    % likelihood factor GPS
    if add_gps
        lambda_gps = H_gps' * (inv(gps_cov)) * H_gps;
        eta_gps = H_gps' * (inv(gps_cov)) * gps_measured;
    
        P_new = inv(lambda_m + lambda_v + lambda_gps);
        mu_new = P_new * (eta_m + eta_v + eta_gps); 
    else
        P_new = inv(lambda_m + lambda_v);
        mu_new = P_new * (eta_m + eta_v);
        % P_new = inv(lambda_m );
        % mu_new = P_new * (eta_m);
    end

end