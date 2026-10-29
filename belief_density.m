function [GX, GY, Z, stats] = belief_density(sets, h, lim)
%BELIEF_DENSITY  Kernel density of one or more particle sets, plus the
%                starvation diagnostics. No plotting.
%
%   [GX, GY, Z, stats] = BELIEF_DENSITY(sets, h)
%   [GX, GY, Z, stats] = BELIEF_DENSITY(sets, h, lim)
%
%   sets  cell array of sample sets. Each is N x 6 (a belief, weight in
%         col 5) or N x 4 / N x 3 (a message, weight in col 3).
%   h     the bandwidth you actually use in product_messages
%   lim   optional [xmin xmax ymin ymax]; computed from the data if absent
%
%   Z is the SUM of the sets' densities, so a single set gives that set's
%   density and several give what their union looks like.
%
%   stats is a 1 x numel(sets) struct array, one per set:
%     .N .ess .unique .sd .spacing .h_over_space .ripple .verdict
%
%   The ripple is 4*exp(-2*pi^2*h^2/s^2), from Poisson summation on a
%   lattice of kernels spaced s apart. Read it as
%       h/s >= 1.0  ->  ~0 %     perfectly smooth
%       h/s  = 0.5  ->   3 %     the practical floor
%       h/s  = 0.3  ->  68 %     starved: the density breaks into islands

if ~iscell(sets), sets = {sets}; end
n = numel(sets);

P = cell(1,n);  W = cell(1,n);
for s = 1:n
    B = sets{s};
    switch size(B,2)
        case 6,    P{s} = B(:,1:2); W{s} = B(:,5);
        case {4,3},P{s} = B(:,1:2); W{s} = B(:,3);
        otherwise
            error('belief_density:format', ...
                  ['Set %d has %d columns. Expected 6 (belief: weight in ' ...
                   'col 5) or 4/3 (message: weight in col 3).'], s, size(B,2));
    end
    W{s} = W{s} / sum(W{s});
end

% ---- diagnostics, per set ----------------------------------------------
stats = struct('N',{},'ess',{},'unique',{},'sd',{},'spacing',{}, ...
               'h_over_space',{},'ripple',{},'verdict',{});
for s = 1:n
    st.N      = size(P{s},1);
    st.ess    = 1 / sum(W{s}.^2);
    Pu        = unique(P{s}, 'rows');
    st.unique = size(Pu,1);
    st.sd     = std(P{s}, 0, 1);

    % median nearest-neighbour distance among DISTINCT particles: the gap
    % the kernels actually have to bridge. Subsampled so it stays fast.
    m   = min(st.unique, 500);
    sub = Pu(randperm(st.unique, m), :);
    d2  = (sub(:,1) - Pu(:,1).').^2 + (sub(:,2) - Pu(:,2).').^2;
    d2(d2 < eps) = Inf;
    st.spacing = median(sqrt(min(d2, [], 2)));

    st.h_over_space = h / st.spacing;
    st.ripple       = 4 * exp(-2*pi^2 * st.h_over_space^2);
    if     st.ripple > 0.5,  st.verdict = 'STARVED';
    elseif st.ripple > 0.05, st.verdict = 'lumpy';
    else,                    st.verdict = 'ok';
    end
    stats(s) = st;                                                  %#ok<AGROW>
end

% ---- grid ---------------------------------------------------------------
allP = vertcat(P{:});
if nargin < 3 || isempty(lim)
    pad = max(3*max(std(allP,0,1)), 6*h);
    lim = [min(allP(:,1))-pad, max(allP(:,1))+pad, ...
           min(allP(:,2))-pad, max(allP(:,2))+pad];
end
ng = 150;
[GX, GY] = meshgrid(linspace(lim(1),lim(2),ng), linspace(lim(3),lim(4),ng));
q = [GX(:), GY(:)];

Z = zeros(size(q,1),1);
for s = 1:n
    blk = max(1, floor(3e6 / size(P{s},1)));
    for a = 1:blk:size(q,1)
        b = min(a+blk-1, size(q,1));
        dd = (q(a:b,1) - P{s}(:,1).').^2 + (q(a:b,2) - P{s}(:,2).').^2;
        Z(a:b) = Z(a:b) + exp(-dd/(2*h^2)) * W{s};
    end
end
Z = reshape(Z, size(GX)) / (2*pi*h^2);
end
