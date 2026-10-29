function plot_spawn_debug(dbg, particle_h, k, t, x_true_state_global, index)
%PLOT_SPAWN_DEBUG  Draw one figure for one SPAWN step from a recorded snapshot.
%
%   plot_spawn_debug(dbg, particle_h, k, tvec(k), x_true_state_global, index)
%
%   This exists so the main script carries no debug branching. The SPAWN loop
%   records three plain assignments:
%
%       dbg.pred     = bel_agents;                  % before the iterations
%       dbg.msg{f,i} = message_agent;               % what drone i received
%       dbg.bel{f}   = bel_agents;                  % after iteration f
%
%   They cost nothing: MATLAB cells are copy-on-write and bel_agents is
%   replaced wholesale each iteration rather than modified in place, so these
%   hold references, not copies. Peak extra memory is the few generations of
%   particle sets still referenced -- a handful of MB at 4000 particles.
%
%   Columns are drones, rows are stages. Read DOWN a column to follow one
%   drone through the correction, ACROSS a row to compare drones at the same
%   stage.

n = numel(dbg.pred);
F = numel(dbg.bel);

truths = zeros(2, n);
for i = 1:n
    truths(:,i) = x_true_state_global{i}(1:2, index);
end

snap = struct('title', {}, 'sets', {});

% row 1: what the prediction step left us with
row = cell(1, n);
for i = 1:n, row{i} = {dbg.pred{i}}; end
snap(end+1) = struct('title', 'after prediction', 'sets', {row});

% row 2: the incoming rings of the first iteration, before the product
if F >= 1 && ~isempty(dbg.msg)
    snap(end+1) = struct('title', 'incoming messages', 'sets', {dbg.msg(1,:)});
end

% one row per SPAWN iteration: the belief after the product
for f = 1:F
    if isempty(dbg.bel{f}), continue; end
    row = cell(1, n);
    for i = 1:n, row{i} = {dbg.bel{f}{i}}; end
    snap(end+1) = struct('title', sprintf('after SPAWN iter %d', f), 'sets', {row}); %#ok<AGROW>
end

plot_spawn_step(snap, particle_h, k, t, truths);
end
