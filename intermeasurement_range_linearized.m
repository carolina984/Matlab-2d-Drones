function [mu_new, P_new] = intermeasurement_range_linearized(mu_j, P_j, mu_i, P_i, sigma_range, range_measurement)
    z = range_measurement;
    R = sigma_range^2;
    
    lambda_m = inv(P_i);
    eta_m = lambda_m * mu_i;
    
    H = [1 0 0 0;
          0 1 0 0];
    
    A = ((H * (mu_j-mu_i)).' * H) / norm(H * (mu_j-mu_i));
    z_tilde = z - norm(H * (mu_j-mu_i)) - A * mu_i;
    cov = A * P_j * A.' + R;
    
    lambda_r = A.' * inv(cov) * A;
    eta_r = -A.' * inv(cov) * z_tilde;

    lambda = lambda_m + lambda_r;
    eta = eta_m + eta_r;
    
    P_new = inv(lambda);
    mu_new = P_new * (eta);
end