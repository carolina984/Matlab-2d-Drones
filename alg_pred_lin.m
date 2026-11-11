function alg = alg_pred_lin()
%ALG_EKF  Gaussian belief, no cooperation.
alg.name     = 'EKF';
alg.init     = @init;
alg.predict  = @predict;
alg.correct  = @(S, meas, k, cfg) S;        % no cooperation: do nothing
alg.estimate = @estimate;
end

function s = init(cfg, x0)
s.mu = x0;
s.P  = cfg.P_0;
end

function s = predict(s, meas, i, k, cfg)
u = meas.u(:,i,k);
Q = cfg.sigma_process^2 * eye(4);

[s.P, s.mu] = prediction_operation_with_angles(0, 0, s.mu, s.P, ...
    meas.theta(i,k), cfg.sigma_theta, meas.vel(:,i,k), u, cfg.A, cfg.B, Q, ...
    cfg.H_vel, cfg.V_cov, meas.gps(:,i,k), meas.gps_on(k,i), cfg.gps_cov);
end

function mu = estimate(s)
mu = s.mu;
end
