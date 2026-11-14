function [belief, ess, mu_new, P_new] = prediction_operation_particles(belief, theta, ...
            sigma_theta, vel_measured, u, A, B, Q_cov, V_cov, ...
            gps_measured, add_gps, gps_cov)
%PREDICTION_OPERATION_PARTICLES  Particle twin of prediction_operation_with_angles.
%
%   [belief, ess, mu_new, P_new] = prediction_operation_particles(...)
%
%   belief : N x 6  [x y vx vy w informative], global frame, w sums to 1
%   mu_new : 4 x 1  weighted mean of the cloud   <-- compare with the Gaussian
%   P_new  : 4 x 4  weighted covariance          <-- compare with the Gaussian
%
%   Written to mirror prediction_operation_with_angles block for block, with
%   the same names (R_, Rt_, dR_, H_gps, lambda_*, eta_*, mu_new, P_new) and
%   the same section order, so the two can be read side by side.
%
%   TO COMPARE THEM DIRECTLY
%       [bel, ess, mu_p, P_p] = prediction_operation_particles(bel, theta, ...);
%       [P_g, mu_g]           = prediction_operation_with_angles(idx, i, mu, P, theta, ...);
%       norm(mu_p - mu_g)                 % should be small
%       norm(P_p - P_g) / norm(P_g)       % will NOT be small -- see below
%
%   ======================================================================
%   WHERE THE TWO AGREE, AND WHERE THEY DO NOT
%   ======================================================================
%   SAME: the rotation conventions. Both use Rt_ = [c s; -s c] for
%         body -> global on the acceleration, and R_ = [c -s; s c] for
%         global -> body on the velocity measurement. (That is a
%         CLOCKWISE-POSITIVE heading; set heading_sign = -1 in the
%         trajectory generator to match.)
%   SAME: the state transition A, the input matrix B, the process noise
%         Q_cov, and the GPS measurement model H_gps with gps_cov.
%
%   DIFFERENT 1 -- how sigma_theta enters. THIS IS THE MAIN ONE.
%         The Gaussian LINEARISES. It carries the heading uncertainty as two
%         separate first-order terms:
%             (B*dRt_*u)*sigma_theta^2*(u'*dR_*B')     added to P
%             dR_*H_vel*mu*sigma_theta^2*mu'*H_vel'*dRt_   inside C
%         The particle filter SAMPLES: th = theta + sigma_theta*randn(N,1),
%         one draw per particle, used for BOTH the acceleration rotation and
%         the velocity prediction. No Jacobian, exact to all orders.
%         They agree to first order in sigma_theta and diverge at
%         O(sigma_theta^2) ~ 4% at your sigma_theta = 0.2.
%
%   DIFFERENT 2 -- correlation between the two heading effects.
%         One heading error rotates the acceleration AND biases the predicted
%         velocity measurement. The particle filter uses the SAME th for
%         both, so that correlation is captured. The Gaussian adds the two
%         terms independently (one to P, one to C) and so neglects it.
%
%   DIFFERENT 3 -- where the Jacobian is evaluated.
%         The Gaussian's C uses mu_old, the predicted MEAN. The particle
%         filter uses each particle's own velocity. Standard EKF-vs-PF.
%
%   DIFFERENT 4 -- GPS precision floor.
%         Algebraically identical models, but a particle cloud of sd s with
%         N particles has spacing s*sqrt(2*pi/N) and cannot exploit a
%         measurement sharper than that. At s = 1 m, N = 2000 the spacing is
%         0.056 m, so gps_std below ~0.2 m collapses the cloud (ESS -> 1)
%         while the Gaussian just sets the position to the reading. Measured.
%
%   DIFFERENT 5 -- resampling. No Gaussian analogue; see the last block.
%   ======================================================================

N = size(belief, 1);

H_gps = [1 0 0 0;
         0 1 0 0];
H_vel = [0 0 1 0;
         0 0 0 1];

% ---- rotation matrices --------------------------------------------------
% GAUSSIAN: one R_ at the measured theta, plus derivative matrices dR_, dRt_
%           used to propagate sigma_theta to first order.
% PARTICLE: one rotation PER PARTICLE at a SAMPLED theta. The derivative
%           matrices have no counterpart because nothing is linearised --
%           that is Difference 1.
th = theta + sigma_theta * randn(N, 1);
c  = cos(th);   s = sin(th);
% per particle:  Rt_ = [c s; -s c]  (body->global),  R_ = [c -s; s c] (global->body)

%% ---- prediction operation ---------------------------------------------
% GAUSSIAN:  mu_old = A*mu_old + B*(Rt_*u);
%            P_old  = A*P_old*A' + (B*dRt_*u)*sigma_theta^2*(u'*dR_*B') + Q_cov;
% PARTICLE:  the same mean update applied to every particle, with that
%            particle's own Rt_. The heading term in P has no counterpart:
%            its effect is already in the spread of the propagated cloud.
X = belief(:, 1:4);
lw = log(max(belief(:, 5), realmin));

a_gx =  c*u(1) + s*u(2);                    % Rt_*u, per particle
a_gy = -s*u(1) + c*u(2);

X = X * A.' + [a_gx, a_gy] * B.';           % A*x + B*(Rt_*u)
X = X + randn(N, 4) * chol(Q_cov, 'lower').';   % + Q_cov

% ---- motion model factor ------------------------------------------------
% GAUSSIAN:  lambda_m = inv(P_old);  eta_m = lambda_m * mu_old;
% PARTICLE:  the propagated cloud {X, lw} IS this factor. There is no
%            canonical form to convert to -- the samples carry it.

%% ---- likelihood factor VELOCITY ----------------------------------------
% GAUSSIAN:  C        = dR_*H_vel*mu_old*sigma_theta^2*mu_old'*H_vel'*dRt_
%                       + R_*V_cov*Rt_;
%            lambda_v = H_vel' * Rt_ * inv(C) * R_ * H_vel;
%            eta_v    = H_vel' * Rt_ * inv(C) * vel_measured;
% PARTICLE:  predicted measurement is R_ * v_global per particle, residual
%            against vel_measured, weighted by V_cov ALONE. The sigma_theta
%            inflation of C is absent BECAUSE the heading is already sampled
%            -- adding it here would double-count. That is Difference 1
%            seen from the measurement side.
vb_x = c.*X(:,3) - s.*X(:,4);               % R_ * v_global, per particle
vb_y = s.*X(:,3) + c.*X(:,4);

rv = [vel_measured(1) - vb_x, vel_measured(2) - vb_y];
lw = lw - 0.5 * sum((rv / V_cov) .* rv, 2);

%% ---- likelihood factor GPS ---------------------------------------------
% GAUSSIAN:  lambda_gps = H_gps' * inv(gps_cov) * H_gps;
%            eta_gps    = H_gps' * inv(gps_cov) * gps_measured;
% PARTICLE:  the same Gaussian likelihood, as a log-weight increment.
%            Same model; see Difference 4 for why the behaviour differs.
if nargin >= 11 && add_gps
    rg = [gps_measured(1) - X(:,1), gps_measured(2) - X(:,2)];
    lw = lw - 0.5 * sum((rg / gps_cov) .* rg, 2);
end

%% ---- combine the factors -----------------------------------------------
% GAUSSIAN:  P_new  = inv(lambda_m + lambda_v + lambda_gps);
%            mu_new = P_new * (eta_m + eta_v + eta_gps);
% PARTICLE:  adding log-weights IS the product of the factors. Normalise in
%            the log domain so no factor underflows before the others are in.
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

%% ---- mu_new and P_new, for comparison with the Gaussian ---------------
% The same two quantities the Gaussian returns, read off the weighted cloud.
mu_new = (w.' * X).';                                   % 4 x 1
dX     = X - mu_new.';
P_new  = (dX .* w).' * dX;                              % 4 x 4
P_new  = 0.5 * (P_new + P_new.');                       % symmetrise

%% ---- resampling (NO GAUSSIAN COUNTERPART) ------------------------------
% Liu & Chen (1998): resample once the effective sample size falls below
% N/2. The Gaussian has nothing like this because it never degenerates --
% which is exactly why a particle/Gaussian comparison is not a comparison
% of two estimators of the same thing unless you report ESS alongside.
ess = 1 / sum(w.^2);
if ess < N/2
    idx    = randsample(N, N, true, w);
    belief = [X(idx,:), ones(N,1)/N, belief(idx,6)];
else
    belief = [X, w, belief(:,6)];
end

end
