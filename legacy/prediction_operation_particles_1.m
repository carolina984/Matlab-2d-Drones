
function [particles_new, weights_new, mu_new, P_new] = ...
    prediction_operation_particles_1( ...
    particles_old, weights_old, theta, sigma_theta, ...
    vel_measured, u, A, B, Q, H_vel, V_cov, ...
    gps_measured, add_gps)

    % PARTICLE PREDICTION FOR SPAWN
    %
    % State: [px; py; vx; vy]
    % particles_old: N x 4
    % weights_old:   N x 1, normalized
    %
    % The prediction incorporates:
    %   1. Motion model
    %   2. Body-frame velocity likelihood
    %   3. Optional GPS likelihood
    %
    % Relative UAV measurements belong in the SPAWN
    % correction operation, not in this function.

    N = size(particles_old, 1);

    weights_old = weights_old(:);
    weights_old = weights_old / sum(weights_old);

    % Process-noise square root
    LQ = chol(Q, 'lower');

    % Heading rotation matrices
    R = @(th) [cos(th), -sin(th);
               sin(th),  cos(th)];

    % Store predicted particles and their log weights
    particles_new = zeros(N, 4);
    log_weights = -inf(N, 1);

    % Measurement covariance inverses
    % Use linear solves rather than explicit inv().
    LV = chol(V_cov, 'lower');

    if add_gps
        sigma_gps = 2;
        gps_cov = sigma_gps^2 * eye(2);
        LG = chol(gps_cov, 'lower');
    end

    for n = 1:N

        % 1. Sample uncertain heading for this particle
        theta_n = theta + sigma_theta * randn;

        R_n = R(theta_n);
        Rt_n = R_n';

        % 2. Propagate state through motion model
        x_old = particles_old(n, :)';

        process_noise = LQ * randn(4, 1);

        x_new = A * x_old ...
              + B * (Rt_n * u) ...
              + process_noise;

        particles_new(n, :) = x_new';

        % Start with the previous particle weight
        if weights_old(n) <= 0
            continue;
        end

        log_w = log(weights_old(n));

        % 3. Velocity measurement likelihood
        % Model: vel_measured = R(theta) * H_vel*x + noise
        vel_predicted = R_n * H_vel * x_new;

        residual_vel = vel_measured - vel_predicted;

        whitened_vel = LV \ residual_vel;

        log_w = log_w ...
              - 0.5 * (whitened_vel' * whitened_vel);

        % 4. Optional GPS likelihood
        if add_gps

            H_gps = [1 0 0 0;
                     0 1 0 0];

            gps_predicted = H_gps * x_new;
            residual_gps = gps_measured - gps_predicted;

            whitened_gps = LG \ residual_gps;

            log_w = log_w ...
                  - 0.5 * (whitened_gps' * whitened_gps);
        end

        log_weights(n) = log_w;
    end

    % 5. Normalize weights numerically stably
    valid = isfinite(log_weights);

    if ~any(valid)
        error('All particle weights are zero or invalid.');
    end

    max_log_w = max(log_weights(valid));

    weights_new = zeros(N, 1);
    weights_new(valid) = exp( ...
        log_weights(valid) - max_log_w);

    weights_new = weights_new / sum(weights_new);

    % 6. Weighted mean and covariance for diagnostics
    mu_new = (weights_new' * particles_new)';

    deviations = particles_new - mu_new';

    P_new = deviations' * ...
        (deviations .* weights_new);

end
