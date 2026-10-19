function [belief] = gaussian_belief(mu, P, number_samples)
% BELIEF_ESTIMATE  MMSE estimate and covariance of a weighted particle set.
%
%   belief : n x 6  [x, y, vx, vy, w, informative]
%   mu     : 1 x 4  weighted mean          (the MMSE estimate)
%   P      : 4 x 4  weighted covariance    (what the filter CLAIMS to know)
%   sigma  : scalar per-axis std, sqrt(trace(P)/2)  -- feeds the calibration ratio

% Initial Gaussian belief
L = chol(P, 'lower');

Z = randn(4, number_samples);

particles = (mu + L * Z).';

belief = [particles, ones(number_samples, 1)/number_samples, 1*ones(number_samples, 1)]; 

end
