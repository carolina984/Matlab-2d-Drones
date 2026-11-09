function [belief] = resampling_operation(belief)
%% ---- resampling (NO GAUSSIAN COUNTERPART) ------------------------------
% Liu & Chen (1998): resample once the effective sample size falls below
% N/2. The Gaussian has nothing like this because it never degenerates --
% which is exactly why a particle/Gaussian comparison is not a comparison
% of two estimators of the same thing unless you report ESS alongside.

N = size(belief, 1);
ess = 1 / sum(belief(:,5).^2);
if ess < N/2
    idx    = randsample(N, N, true, w);
    belief = [belief(idx,1:4), ones(N,1)/N, belief(idx,6)];
else
    belief = [belief(:, 1:4), belief(:,5), belief(:,6)];
end

% fprintf("i resampled!")
end