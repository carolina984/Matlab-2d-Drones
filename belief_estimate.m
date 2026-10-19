function [mu, P, sigma] = belief_estimate(belief)
% BELIEF_ESTIMATE  MMSE estimate and covariance of a weighted particle set.
%
%   belief : n x 4  [x, y, w, informative]
%   mu     : 1 x 2  weighted mean          (the MMSE estimate)
%   P      : 2 x 2  weighted covariance    (what the filter CLAIMS to know)
%   sigma  : scalar per-axis std, sqrt(trace(P)/2)  -- feeds the calibration ratio

x = belief(:, 1:4);
w = belief(:, 5);
w = w / sum(w);

mu = w' * x;                 % 1 x 2
d  = x - mu;
P  = (d .* w)' * d;          % 2 x 2 weighted covariance
sigma = sqrt(trace(P) / 2);
end
