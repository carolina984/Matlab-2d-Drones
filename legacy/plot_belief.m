function stats = plot_belief(belief, h, name, true_pos)
%PLOT_BELIEF  Contour of a particle belief, with the starvation diagnostics.
%
%   stats = PLOT_BELIEF(belief, h)
%   stats = PLOT_BELIEF(belief, h, name)
%   stats = PLOT_BELIEF(belief, h, name, true_pos)
%
%   belief    N x 6  [x y vx vy w flag]   (or N x 4 [x y w flag], a message)
%   h         the bandwidth you are actually using in product_messages
%   name      string for the title, e.g. 'drone 1, k = 40'
%   true_pos  optional 2-vector, drawn as a red star
%
%   Draws the kernel density that product_messages actually sees -- same
%   kernels, same h -- with the particles on top, and prints the numbers
%   that say whether the cloud is starved.
%
%   WHAT TO LOOK FOR
%   healthy : one smooth blob (or two, if genuinely bimodal), particles
%             spread evenly through it
%   starved : the contour breaks into separate islands, each sitting on a
%             clump of particles, with empty space between them. The
%             estimate is then being decided by where the islands happen
%             to fall rather than by the belief.
%
%   The number that predicts it is h / spacing. Kernels spaced s apart
%   with width h produce a relative ripple of
%
%       ripple = 4 * exp(-2*pi^2*h^2/s^2)
%
%   (Poisson summation; verified numerically to four digits.) Read it as:
%       h/s >= 1.0   ripple ~ 1e-8    perfectly smooth
%       h/s  = 0.6   ripple ~ 0.3 %   fine
%       h/s  = 0.5   ripple ~ 3 %     the practical floor
%       h/s  = 0.4   ripple ~ 17 %    lumpy
%       h/s  = 0.3   ripple ~ 68 %    unusable

if nargin < 3 || isempty(name),     name = 'belief'; end
if nargin < 4,                      true_pos = []; end

% ---- read the right columns for a belief (6) or a message (4) ----------
switch size(belief, 2)
    case 6, P = belief(:,1:2); w = belief(:,5);
    case 4, P = belief(:,1:2); w = belief(:,3);
    case 3, P = belief(:,1:2); w = belief(:,3);
    otherwise
        error('plot_belief:format', ...
              'Expected N x 6 (belief) or N x 4/3 (message), got %d columns.', ...
              size(belief,2));
end
N = size(P,1);
w = w / sum(w);

% ---- diagnostics --------------------------------------------------------
stats.N        = N;
stats.ess      = 1 / sum(w.^2);
stats.unique   = size(unique(P, 'rows'), 1);
stats.sd       = std(P, 0, 1);

% median nearest-neighbour distance among the DISTINCT particles -- this is
% the spacing the kernels actually have to bridge. Computed on a subsample
% so it stays fast for large N.
Pu   = unique(P, 'rows');
m    = min(size(Pu,1), 600);
sub  = Pu(randperm(size(Pu,1), m), :);
d2   = (sub(:,1) - Pu(:,1).').^2 + (sub(:,2) - Pu(:,2).').^2;
d2(d2 < eps) = Inf;                       % ignore self-distance
stats.spacing  = median(sqrt(min(d2, [], 2)));

stats.h            = h;
stats.h_over_space = h / stats.spacing;
stats.ripple       = 4 * exp(-2*pi^2 * stats.h_over_space^2);
stats.underflow_m  = h * sqrt(2*709);     % exp() is exactly 0 beyond this

if     stats.ripple > 0.5,  stats.verdict = 'STARVED - contour will break up';
elseif stats.ripple > 0.05, stats.verdict = 'lumpy - h is marginal';
else,                       stats.verdict = 'healthy';
end

% ---- the density product_messages actually sees -------------------------
pad = max(3*max(stats.sd), 6*h);
gx  = linspace(min(P(:,1))-pad, max(P(:,1))+pad, 220);
gy  = linspace(min(P(:,2))-pad, max(P(:,2))+pad, 220);
[GX, GY] = meshgrid(gx, gy);
q   = [GX(:), GY(:)];

Z   = zeros(size(q,1), 1);
blk = max(1, floor(4e6 / N));             % keep the distance matrix small
for a = 1:blk:size(q,1)
    b = min(a+blk-1, size(q,1));
    dd = (q(a:b,1) - P(:,1).').^2 + (q(a:b,2) - P(:,2).').^2;
    Z(a:b) = exp(-dd/(2*h^2)) * w;
end
Z = reshape(Z, size(GX)) / (2*pi*h^2);

% ---- draw ---------------------------------------------------------------
clf; hold on
contourf(GX, GY, Z, 24, 'LineStyle', 'none');
colormap(flipud(bone)); colorbar
scatter(P(:,1), P(:,2), 6, [0.15 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.35);
if ~isempty(true_pos)
    scatter(true_pos(1), true_pos(2), 220, 'p', 'MarkerFaceColor', [0.82 0.29 0.36], ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.6);
end
axis equal tight; grid on; box on
xlabel('x [m]'); ylabel('y [m]')

title({sprintf('%s   —   %s', name, stats.verdict), ...
       sprintf(['N = %d, unique = %d, ESS = %.0f  |  h = %.2f, spacing = %.3f m, ' ...
                'h/s = %.2f, ripple = %.1f%%'], ...
               stats.N, stats.unique, stats.ess, stats.h, stats.spacing, ...
               stats.h_over_space, 100*stats.ripple)}, ...
      'FontWeight', 'normal', 'FontSize', 9);

fprintf('\n--- %s ---\n', name);
fprintf('  particles        %d  (unique %d, ESS %.0f)\n', stats.N, stats.unique, stats.ess);
fprintf('  position sd      %.3f, %.3f m\n', stats.sd(1), stats.sd(2));
fprintf('  kernel spacing   %.3f m\n', stats.spacing);
fprintf('  h / spacing      %.2f   -> ripple %.1f %%\n', stats.h_over_space, 100*stats.ripple);
fprintf('  underflow beyond %.2f m from any kernel\n', stats.underflow_m);
fprintf('  verdict          %s\n', stats.verdict);
end
