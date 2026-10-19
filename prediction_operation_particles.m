function [belief, ess] = prediction_operation_particles(belief, theta, sigma_theta, ...
                            vel_measured, u, A, B, Q_cov, V_cov, ...
                            gps_measured, add_gps, gps_cov)
%PREDICTION_OPERATION_PARTICLES  SPAWN's prediction operation, with particles.
%
%   [BELIEF, ESS] = PREDICTION_OPERATION_PARTICLES(BELIEF, THETA, ...)
%
%   BELIEF is N x 6 :  [x  y  vx  vy  w  informative]
%   Positions and velocities are GLOBAL frame; w sums to 1.
%
%   NO RESAMPLING. Weights are propagated and returned as they are. The
%   only resampling in the whole filter is the one inside the SPAWN
%   correction (product_messages), which is where Ihler's Alg. 2 puts it:
%   draw M from the k*M weighted candidates, "yielding equal-weight
%   samples drawn from the product of all incoming messages".
%
%   WHAT THIS IMPLEMENTS, AND WHOSE IT IS
%   This is SPAWN's prediction operation (Wymeersch, Lien & Win, Alg. 2
%   line 4), not anything of Ihler's -- his sensors are static and his
%   paper has no prediction step at all. SPAWN specifies three factors:
%       incoming message X^(t-1)->h^(t-1)   the previous belief
%       mobility model p(x^t | x^(t-1))     propagate each particle
%       local likelihood p(z_self | x)      weight each particle
%   z_self is the paper's own name for the internal measurements -- your
%   velocity and GPS -- as opposed to z_rel, the inter-node ranges that
%   go to the correction. So the split you already have is the paper's.
%
%   The heading is drawn per particle, theta_s ~ N(theta, sigma_theta^2),
%   which realises the mobility model exactly with no Jacobian. SPAWN
%   leaves the realisation open, so say in the text that this is your
%   choice rather than the paper's.
%
%   ---------------------------------------------------------------
%   READ THIS BEFORE YOU LENGTHEN measurement_interval
%   ---------------------------------------------------------------
%   With no resampling here, the weights compound every step and the only
%   reset is the SPAWN correction. Measured at your parameters (N = 2000,
%   velocity cloud sd 1 m/s, velocity measurement sd 0.4 m/s, dt = 0.5 s),
%   ESS roughly HALVES per predict step:
%
%       step 1   456        step 3   105
%       step 2   201        step 4    54   <- correction resets it
%
%   So the worst ESS over a cycle depends entirely on how many predict
%   steps fit between corrections:
%
%       measurement_interval   predict steps   worst ESS   % of N
%              2 s                   4             54       2.7 %
%              5 s                  10              5       0.2 %
%             10 s                  20              1.4     0.07 %
%
%   At 2 s this is survivable. At 10 s the cloud is dead long before the
%   correction arrives, and every number the filter produces after that
%   is one particle wearing a costume. If you go to a longer interval for
%   the Lambda study, you must either resample here (Liu & Chen 1998,
%   ESS < N/2) or raise N by the same factor you lengthen the interval.
%   Print ESS every step and watch it.

N  = size(belief, 1);
X  = belief(:, 1:4);
lw = log(max(belief(:, 5), realmin));

% ---- mobility model: propagate every particle ---------------------------
% one heading sample per particle -- this is where sigma_theta enters
th = theta + sigma_theta * randn(N, 1);
c  = cos(th);  s = sin(th);

% Rt_ * u, per particle  (body -> global; same convention as your
% Gaussian code, where error_outside = R__*true_acc_local - true_acc_global
% with R__ = Rt_ is what pins the sign down)
a_gx =  c*u(1) + s*u(2);
a_gy = -s*u(1) + c*u(2);

X = X * A.' + [a_gx, a_gy] * B.';          % A*x + B*(Rt_*u), vectorised
X = X + randn(N, 4) * chol(Q_cov, 'lower').';

% ---- local likelihood: velocity, measured in the body frame -------------
% predicted measurement for each particle is R_ * v_global, R_ = Rt_'
vb_x = c.*X(:,3) - s.*X(:,4);
vb_y = s.*X(:,3) + c.*X(:,4);

rv = [vel_measured(1) - vb_x, vel_measured(2) - vb_y];
lw = lw - 0.5 * sum((rv / V_cov) .* rv, 2);

% ---- local likelihood: GPS, when there is one ---------------------------
if nargin >= 11 && add_gps
    rg = [gps_measured(1) - X(:,1), gps_measured(2) - X(:,2)];
    lw = lw - 0.5 * sum((rg / gps_cov) .* rg, 2);
end

% ---- normalise in the log domain, and report ESS ------------------------
lw = lw - max(lw);
w  = exp(lw);
sw = sum(w);

if ~isfinite(sw) || sw <= 0
    warning('prediction_operation_particles:degenerate', ...
            ['Every particle got zero weight. Check the rotation ' ...
             'convention, and that V_cov and gps_cov are covariances ' ...
             'rather than standard deviations.']);
    w = ones(N,1)/N;
else
    w = w / sw;
end

ess = 1 / sum(w.^2);

% belief = [X, w, belief(:, 6)];

idx = randsample(N, N, true, w);
X   = X(idx, :);
belief = [X, ones(N,1)/N, belief(idx, 6)];
end