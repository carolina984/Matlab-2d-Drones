function plot_spawn_step(snap, h, k, t, truths)
%PLOT_SPAWN_STEP  One figure per time step: every stage of SPAWN, stacked.
%
%   PLOT_SPAWN_STEP(snap, h, k, t, truths)
%
%   snap      1 x R struct array, one entry per ROW of the figure:
%               .title  text for the row, e.g. 'after prediction'
%               .sets   1 x num_uavs cell. sets{i} is itself a CELL ARRAY
%                       of sample sets to draw in the panel for drone i.
%                       One set  -> filled contour + particles.
%                       Several  -> particles only, one colour each, so
%                                   you can see the individual rings.
%   h         the bandwidth you use in product_messages
%   k, t      time index and time, for the figure title
%   truths    2 x num_uavs true positions at this step
%
%   Columns are drones, rows are stages, so reading DOWN a column shows
%   what happened to one drone over the correction. Each panel's subtitle
%   carries ESS, h/spacing and the starvation verdict.
%
%   Red star  = the true position of THIS drone (the panel's own node)
%   Grey triangles = the true positions of the other drones (the ring
%                    centres, so you can see whether a message is
%                    centred where it should be)

R = numel(snap);
U = size(truths, 2);

% common axis limits across the whole figure, so rows are comparable
allP = [];
for r = 1:R
    for i = 1:U
        for s = 1:numel(snap(r).sets{i})
            allP = [allP; snap(r).sets{i}{s}(:,1:2)];                %#ok<AGROW>
        end
    end
end
allP = [allP; truths.'];
pad  = 0.08 * max(max(allP,[],1) - min(allP,[],1));   % no toolbox 'range'
lim  = [min(allP(:,1))-pad, max(allP(:,1))+pad, ...
        min(allP(:,2))-pad, max(allP(:,2))+pad];

figure('Name', sprintf('SPAWN step k = %d', k), ...
       'Position', [60 60 360*U 300*R]);
tl = tiledlayout(R, U, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf('SPAWN at k = %d  (t = %.1f s),  h = %.2f', k, t, h), ...
      'FontWeight', 'bold');

cols = [0.16 0.44 0.71; 0.88 0.47 0.20; 0.20 0.62 0.55; 0.55 0.34 0.64];

for r = 1:R
    for i = 1:U
        ax = nexttile; hold(ax, 'on');
        sets = snap(r).sets{i};

        if isempty(sets)
            text(0.5, 0.5, 'no data', 'Units','normalized', ...
                 'HorizontalAlignment','center', 'Color',[.5 .5 .5]);
            sub = '';
        else
            [GX, GY, Z, st] = belief_density(sets, h, lim);

            if numel(sets) == 1
                contourf(ax, GX, GY, Z, 20, 'LineStyle', 'none');
                colormap(ax, flipud(gray));
                scatter(ax, sets{1}(:,1), sets{1}(:,2), 4, cols(1,:), 'filled', ...
                        'MarkerFaceAlpha', 0.30);
                sub = sprintf('ESS %.0f/%d   h/s %.2f   %s', ...
                              st(1).ess, st(1).N, st(1).h_over_space, st(1).verdict);
            else
                for s = 1:numel(sets)
                    c = cols(mod(s-1, size(cols,1))+1, :);
                    scatter(ax, sets{s}(:,1), sets{s}(:,2), 4, c, 'filled', ...
                            'MarkerFaceAlpha', 0.25);
                end
                sub = sprintf('%d messages   h/s %.2f, %.2f   %s', ...
                              numel(sets), st(1).h_over_space, ...
                              st(min(2,end)).h_over_space, st(1).verdict);
            end
        end

        % truth markers
        for j = 1:U
            if j == i
                scatter(ax, truths(1,j), truths(2,j), 230, 'p', ...
                        'MarkerFaceColor', [0.82 0.29 0.36], ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
            else
                scatter(ax, truths(1,j), truths(2,j), 70, '^', ...
                        'MarkerFaceColor', [0.65 0.65 0.65], ...
                        'MarkerEdgeColor', 'k', 'LineWidth', 0.4);
            end
        end

        axis(ax, lim); axis(ax, 'square'); grid(ax, 'on'); box(ax, 'on');
        if r == 1
            title(ax, {sprintf('drone %d', i), sub}, ...
                  'FontWeight','normal','FontSize',9);
        else
            title(ax, sub, 'FontWeight','normal','FontSize',8.5);
        end
        if i == 1
            ylabel(ax, {snap(r).title, 'y [m]'}, 'FontWeight','bold','FontSize',9);
        end
        if r == R, xlabel(ax, 'x [m]'); end
    end
end
drawnow
end
