function [mu_new, P_new] = intermeasurements_lidar(theta, sigma_theta, mu_j, Pj, mu_i, P_i, L_cov, lidar_measurement)
    Hi = [-1 0 0 0;
         0 -1 0 0];
    
    Hj = [1 0 0 0;
          0 1 0 0];
    
    R_ = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    Rt_ = [cos(theta) sin(theta); -sin(theta) cos(theta)];
    dR_ = [0 -1; 1 0] * R_;
    dRt_ = [-sin(theta) cos(theta); -cos(theta) -sin(theta)];
    
    
    % aux = (dR_ * Hj * mu_j +dR_ *Hi * mu_i);
    % C = R_ * Hj * Pj *Hj' *Rt_ + aux * sigma_theta^2 *aux' + R_ * L_cov * Rt_;
    lambda_m = inv(P_i);
    eta_m = lambda_m * mu_i;

    for i = 1:1
        aux = (dR_ * Hj * mu_j +dR_ *Hi * mu_i);
        C = R_ * Hj * Pj *Hj' *Rt_ + aux * sigma_theta^2 *aux' + R_ * L_cov * Rt_;

        lambda_l = Hi' * Rt_ * inv(C) * R_ * Hi;
        eta_l = Hi' * Rt_ * (C \ (lidar_measurement - R_*Hj*mu_j)); %(((lidar_measurement' - (R_ * Hj * mu_j)') * (inv(C)) * R_ * Hi))';
        P_i = inv(lambda_m + lambda_l);
        mu_i= P_i * (eta_m + eta_l);
    end
    
    % lambda_l = Hi' * Rt_ * inv(C) * R_ * Hi;
    % eta_l = Hi' * Rt_ * (C \ (lidar_measurement - R_*Hj*mu_j)); %(((lidar_measurement' - (R_ * Hj * mu_j)') * (inv(C)) * R_ * Hi))';
    
    % lambda_m = inv(P_i);
    % eta_m = lambda_m * mu_i;

    lambda = lambda_m + lambda_l;
    eta = eta_m + eta_l;
    
    P_new = inv(lambda);
    mu_new = P_new * (eta);
end