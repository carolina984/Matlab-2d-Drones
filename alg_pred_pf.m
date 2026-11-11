function alg = alg_pred_pf()
%ALG_PF  Particle belief, no cooperation.
alg.name     = 'PF';
alg.init     = @init;
alg.predict  = @predict;
alg.correct  = @(S, meas, k, cfg) S;        % no cooperation: do nothing
alg.estimate = @estimate;
end

function s = init(cfg, x0)
s.bel   = gaussian_belief(x0, cfg.P_0, cfg.intra_samples);
s.bcast = intrasamples_to_intersamples(s.bel, cfg.inter_samples);
end

function s = predict(s, meas, i, k, cfg)
u = meas.u(:,i,k);
Q = cfg.sigma_process^2 * eye(4);
s.bel = prediction_operation_particles(s.bel, meas.theta(i,k), ...
    cfg.sigma_theta, meas.vel(:,i,k), u, cfg.A, cfg.B, Q, cfg.V_cov, ...
    meas.gps(:,i,k), meas.gps_on(k,i), cfg.gps_cov);

% resample here, before the correction reads the belief as a density
s.bel   = resampling_operation(s.bel);
s.bcast = intrasamples_to_intersamples(s.bel, cfg.inter_samples);
end

function mu = estimate(s)
mu = belief_estimate(s.bel);
mu = mu(:);
end
