function [P_new, mu_new] = prediction_operation(mu_old,P_old, vel_measured, u, A, B, Q_cov, H_vel, V_cov, gps_measured , add_gps)
    sigma_gps = 2;
    gps_noise = sigma_gps^2 * eye(2);
    gps_cov = gps_noise;
    H_gps = [1 0 0 0;
        0 1 0 0];

    % prediction operation
    mu_old = A*mu_old + B*u;
    P_old  = A*P_old*A' + Q_cov;

    % motion model factor (converted to canonical form)
    lambda_m = inv(P_old);
    eta_m = lambda_m * mu_old;

    % % likelihood factor VELOCITY
    lambda_v = H_vel' * (inv(V_cov)) * H_vel;
    eta_v = H_vel' * (inv(V_cov)) * vel_measured;
    % 
    % % likelihood factor GPS
    lambda_gps = H_gps' * (inv(gps_cov)) * H_gps;
    eta_gps = H_gps' * (inv(gps_cov)) * gps_measured;
    % 
    % 
    % fusion of the factors (converted to moments form)
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