function lp = log_evaluate_all_points(q, mixture, h)
%LOG_EVALUATE_ALL_POINTS  log of a Gaussian kernel mixture, evaluated at q.
%
%   lp = LOG_EVALUATE_ALL_POINTS(q, mixture, h)
%
%   q       : Nq x 2   query points (extra columns are ignored)
%   mixture : M  x 3   [x  y  w]  kernel centres and their weights
%   h       : scalar   kernel bandwidth
%   lp      : Nq x 1   log of
%
%       p(q_a) = SUM_m  w_m * exp(-||q_a - x_m||^2 / (2h^2)) / (2*pi*h^2)
%
%   Drop-in replacement for evaluate_all_points. It returns the LOG of the
%   same number, which is the only reason it exists.
%
%   WHY THE LOG
%   exp(x) in binary64 is exactly 0 for x < -745.13. The linear version
%   therefore returns a hard 0 once a query point is further than
%   h*sqrt(2*745.13) from every centre, and product_messages multiplies
%   J+1 of these together, so the whole product dies at
%
%       d_rms  >  h * sqrt(2*745.13/(J+1))
%
%   = 5.79 m for J = 3 and 4.38 m for J = 6, at h = 0.3. Past that point
%   prod_v./sum_v is 0/0 = NaN and randsample throws "W must contain
%   non-negative values". Note the threshold gets WORSE as J grows: more
%   neighbours should mean more information, so a failure that deepens with
%   J is the signature of the arithmetic failing, not of the inference
%   failing. In the log domain the product is a sum and nothing underflows.
%
%   WHAT THIS DOES AND DOES NOT CHANGE
%   Where the linear version survives, the two agree to 1e-15 and the
%   posterior estimates are identical -- the candidates that underflow have
%   true weights below exp(-745) relative to the best one, so discarding
%   them really does change nothing. This is a guard, not a correction. Its
%   value is that the degenerate case (beliefs and ranges that no longer
%   agree anywhere, so every candidate is hopeless) now comes back as
%   ESS = 1 plus a warning you can trace, instead of an exception.
%
%   TRUNCATION
%   A centre TRUNC bandwidths away contributes exp(-TRUNC^2/2) relative to
%   one at zero distance: 1.5e-8 at TRUNC = 6. So only the nearest centres
%   matter, and knnsearch finds those with a kd-tree instead of forming the
%   whole Nq x M distance matrix. NK is doubled until the furthest
%   neighbour kept is itself past the truncation radius, which is the test
%   for "nothing that mattered was left out". The test does real work here:
%   a ring message is sparse and settles at NK = 256, while the prediction
%   blob is dense and grows until it is effectively exact. If knnsearch is
%   not on the path the dense branch gives the same answer in chunks of
%   bounded memory.

TRUNC = 6;        % keep centres within this many bandwidths
NK0   = 256;      % first guess at how many neighbours that takes
CHUNK = 512;      % dense branch: query rows per block

persistent haveKNN
if isempty(haveKNN)
    haveKNN = exist('knnsearch', 'file') ~= 0;
end

X  = mixture(:, 1:2);
w  = mixture(:, 3);
M  = size(X, 1);
q  = q(:, 1:2);
Nq = size(q, 1);

lw   = log(w);              % log(0) = -Inf, which is the right answer
lnrm = log(2*pi*h^2);

if haveKNN && M > NK0
    % ---- sparse branch: only the nearest centres can contribute --------
    NK = NK0;
    [idx, dst] = knnsearch(X, q, 'K', NK);
    while NK < M && any(dst(:,end) < TRUNC*h)
        NK = min(2*NK, M);                    % the furthest neighbour kept
        [idx, dst] = knnsearch(X, q, 'K', NK);  % was still close enough to count
    end
    lp = logsumexp_rows(lw(idx) - dst.^2/(2*h^2)) - lnrm;
else
    % ---- dense branch: exact, chunked so the memory stays bounded ------
    lp = zeros(Nq, 1);
    for a = 1:CHUNK:Nq
        b  = min(a + CHUNK - 1, Nq);
        d2 = (q(a:b,1) - X(:,1).').^2 + (q(a:b,2) - X(:,2).').^2;
        lp(a:b) = logsumexp_rows(lw.' - d2/(2*h^2));
    end
    lp = lp - lnrm;
end
end

% -------------------------------------------------------------------------
function s = logsumexp_rows(L)
%LOGSUMEXP_ROWS  log(sum(exp(L), 2)), without overflow or underflow.
%
% Shifting each row by its own maximum makes the largest term exp(0) = 1,
% so the sum is never 0 and never Inf. A row that is all -Inf -- every
% centre has zero weight, or every centre is unreachably far -- has a true
% log-density of -Inf, and is returned as that rather than as NaN. Leaving
% it as NaN is how 0/0 gets into the weights in the first place.

t     = max(L, [], 2);
empty = ~isfinite(t);
t(empty) = 0;                   % keep the arithmetic finite ...
s = t + log(sum(exp(L - t), 2));
s(empty) = -Inf;                % ... then say what the answer really is
end
