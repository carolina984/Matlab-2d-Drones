function cfg = generate_swarm_trajectory(cfg)
%GENERATE_SWARM_TRAJECTORY  Build a formation-flying ground truth for SPAWN.
%
%   cfg = GENERATE_SWARM_TRAJECTORY()            % all defaults
%   cfg = GENERATE_SWARM_TRAJECTORY(cfg)         % override any field
%
%   Writes the .mat files your spawn_angles_particles.m loads, in exactly
%   the same format as your original generator:
%
%     x_true_state_global{i}  5 x N   [x; y; vx; vy; t]
%     true_velocity_local{i}  3 x N   [vx_body; vy_body; t]
%     true_acc_global{i}      3 x N   [ax; ay; t]
%     true_acc_local{i}       3 x N   [ax_body; ay_body; t]
%     true_theta{i}           1 x N
%     sim_data                [T; dt_sim; num_uavs]
%
%   HOW IT WORKS
%   One smooth CENTROID path is built from waypoints with minjerkpolytraj.
%   The formation is then carried along it as a rigid body: each drone sits
%   at a fixed offset in the centroid's frame, so the whole shape rotates
%   with the direction of travel. Per-drone velocity and acceleration come
%   from differentiating the resulting positions, which keeps the rotation
%   terms (centripetal, Euler) exact without hand-deriving them.
%
%   ---------------------------------------------------------------
%   READ THIS: THE HEADING SIGN CONVENTION
%   ---------------------------------------------------------------
%   Your filter converts body to global with
%       a_gx =  c*u(1) + s*u(2);
%       a_gy = -s*u(1) + c*u(2);
%   which is the matrix [c s; -s c]. That is the standard GLOBAL-to-BODY
%   rotation, so using it in the body-to-global direction only works if
%   theta is the NEGATIVE of the heading. Your original generator had the
%   mirror-image error (it applied body-to-global to a global vector), so
%   the two cancelled and the round trip came out right -- but the
%   intermediate "local" frame was not the physical body frame, and theta
%   was not the heading.
%
%   cfg.theta_convention picks what to do:
%
%     'legacy'   (default) theta = -heading. Matches your filter exactly as
%                it stands. Nothing else needs changing, and every number
%                the filter produces is correct. Say in the thesis that
%                theta is defined with the opposite sign to the usual
%                heading, or quietly switch to 'standard'.
%
%     'standard' theta = heading, the physical definition. Requires ONE
%                sign change in each of prediction_operation_particles.m
%                and prediction_operation_with_angles.m:
%                    a_gx = c*u(1) - s*u(2);
%                    a_gy = s*u(1) + c*u(2);
%                and the same flip wherever R__ is used in the main script.
%                Preferable for a thesis, because then theta IS the heading
%                and the observability argument reads honestly.
%
%   Requires minjerkpolytraj (Robotics System Toolbox / Navigation Toolbox).

% ======================= defaults =======================================
if nargin < 1, cfg = struct(); end
D = struct( ...
    'T',            400, ...        % duration [s]
    'dt_sim',       0.01, ...       % truth sampling step [s]
    'formation',    'hex7', ...     % 'hex7' 'pent6' 'grid9' 'tri3' or n x 2
    'spacing',      15, ...         % nearest-neighbour spacing [m]
    'path',         'lawnmower', ...% 'lawnmower' 'spiral' 'racetrack' 'waypoints'
    'speed',        12, ...         % nominal ground speed [m/s]
    'lm_leg',       900, ...        % lawnmower: length of a straight leg [m]
    'lm_lanes',     4, ...          % lawnmower: number of legs
    'lm_lane_sep',  250, ...        % lawnmower: spacing between legs [m]
    'sp_r0',        150, ...        % spiral: starting radius [m]
    'sp_growth',    120, ...        % spiral: radius gained per full turn [m]
    'sp_turns',     2.5, ...        % spiral: number of turns
    'rt_len',       800, ...        % racetrack: straight length [m]
    'rt_rad',       250, ...        % racetrack: turn radius [m]
    'wp',           [], ...         % 2 x M custom centroid waypoints
    'body_fixed',   true, ...       % formation rotates with heading
    'breathe_amp',  0, ...          % formation scale oscillation, 0..0.5
    'breathe_T',    120, ...        % breathing period [s]
    'theta_convention', 'legacy', ...
    'R_comm',       100, ...        % ranging radius, for the diagnostics only
    'save_dir',     '.', ...
    'plots',        true);
f = fieldnames(D);
for a = 1:numel(f)
    if ~isfield(cfg, f{a}), cfg.(f{a}) = D.(f{a}); end
end

tvec_sim  = 0:cfg.dt_sim:cfg.T;
steps_sim = numel(tvec_sim);

% ======================= formation offsets (body frame) =================
s = cfg.spacing;
if isnumeric(cfg.formation)
    off = cfg.formation;
else
    switch lower(cfg.formation)
        case 'hex7'                                   % centre + ring of 6
            off = [0 0; s*cos((0:5)*pi/3).', s*sin((0:5)*pi/3).'];
        case 'pent6'                                  % centre + ring of 5
            off = [0 0; s*cos((0:4)*2*pi/5).', s*sin((0:4)*2*pi/5).'];
        case 'grid9'
            [gx, gy] = meshgrid((-1:1)*s, (-1:1)*s);
            off = [gx(:), gy(:)];
        case 'tri3'
            off = s*[0 0; 1 0; 0.5 sqrt(3)/2];  off = off - mean(off,1);
        otherwise
            error('generate_swarm_trajectory:formation', ...
                  'Unknown formation "%s".', cfg.formation);
    end
end
off = off - mean(off, 1);              % centre it on the centroid
num_uavs = size(off, 1);
cfg.offsets  = off;
cfg.num_uavs = num_uavs;

% ======================= centroid waypoints =============================
switch lower(cfg.path)
    case 'lawnmower'
        wp = [];
        for L = 0:cfg.lm_lanes-1
            x0 = 0;  x1 = cfg.lm_leg;
            if mod(L,2), tmp = x0; x0 = x1; x1 = tmp; end
            wp = [wp, [x0; L*cfg.lm_lane_sep], [x1; L*cfg.lm_lane_sep]];   %#ok<AGROW>
        end
    case 'spiral'
        a  = linspace(0, 2*pi*cfg.sp_turns, max(8, round(12*cfg.sp_turns)));
        r  = cfg.sp_r0 + cfg.sp_growth*a/(2*pi);
        wp = [r.*cos(a); r.*sin(a)];
    case 'racetrack'
        a  = linspace(0, 2*pi, 25);
        wp = [cfg.rt_len/2*sign(cos(a)).*abs(cos(a)).^0.6; ...
              cfg.rt_rad      *sin(a)];
    case 'waypoints'
        assert(~isempty(cfg.wp), 'cfg.wp must be 2 x M when path = ''waypoints''.');
        wp = cfg.wp;
    otherwise
        error('generate_swarm_trajectory:path', 'Unknown path "%s".', cfg.path);
end

% time each leg so the centroid holds roughly cfg.speed, then rescale to T
seg  = [0, cumsum(vecnorm(diff(wp,1,2), 2, 1))];
tpts = seg / max(seg(end), eps) * cfg.T;
cfg.path_length = seg(end);
cfg.mean_speed  = seg(end) / cfg.T;

% ======================= smooth centroid ================================
[c_pos, c_vel, ~] = minjerkpolytraj(wp, tpts, steps_sim);

% ======================= heading ========================================
psi = unwrap(atan2(c_vel(2,:), c_vel(1,:)));      % the PHYSICAL heading
switch lower(cfg.theta_convention)
    case 'legacy',   theta =  -psi;
    case 'standard', theta =   psi;
    otherwise, error('theta_convention must be ''legacy'' or ''standard''.');
end
cfg.heading = psi;

% ======================= carry the formation ============================
breathe = 1 + cfg.breathe_amp * sin(2*pi*tvec_sim/cfg.breathe_T);

x_true_state_global = cell(1, num_uavs);
true_velocity_local = cell(1, num_uavs);
true_acc_global     = cell(1, num_uavs);
true_acc_local      = cell(1, num_uavs);
true_theta          = cell(1, num_uavs);

for i = 1:num_uavs
    if cfg.body_fixed
        cp = cos(psi);  sp = sin(psi);                    % body -> global
        ox = off(i,1)*breathe;  oy = off(i,2)*breathe;
        px = c_pos(1,:) + cp.*ox - sp.*oy;
        py = c_pos(2,:) + sp.*ox + cp.*oy;
    else
        px = c_pos(1,:) + off(i,1)*breathe;
        py = c_pos(2,:) + off(i,2)*breathe;
    end

    % differentiate the actual path: picks up the centripetal and Euler
    % terms from the rotating formation without deriving them by hand
    vx = gradient(px, cfg.dt_sim);   vy = gradient(py, cfg.dt_sim);
    ax = gradient(vx, cfg.dt_sim);   ay = gradient(vy, cfg.dt_sim);

    % global -> body, with theta per the chosen convention
    ct = cos(theta);  st = sin(theta);
    vbx =  ct.*vx - st.*vy;      vby =  st.*vx + ct.*vy;
    abx =  ct.*ax - st.*ay;      aby =  st.*ax + ct.*ay;

    x_true_state_global{i} = [px; py; vx; vy; tvec_sim];
    true_velocity_local{i} = [vbx; vby; tvec_sim];
    true_acc_global{i}     = [ax;  ay;  tvec_sim];
    true_acc_local{i}      = [abx; aby; tvec_sim];
    true_theta{i}          = theta;
end

sim_data = [cfg.T; cfg.dt_sim; num_uavs];

% ======================= save ===========================================
d = cfg.save_dir;
save(fullfile(d,'x_true_state_global.mat'), 'x_true_state_global');
save(fullfile(d,'true_velocity_local.mat'), 'true_velocity_local');
save(fullfile(d,'true_acc_global.mat'),     'true_acc_global');
save(fullfile(d,'true_acc_local.mat'),      'true_acc_local');
save(fullfile(d,'true_theta.mat'),          'true_theta');
save(fullfile(d,'sim_data_sim1.mat'),       'sim_data');

% ======================= graph diagnostics over time ====================
nchk = 200;
ks   = round(linspace(1, steps_sim, nchk));
deg_min = zeros(1,nchk); deg_mean = zeros(1,nchk);
globrig = false(1,nchk); tpe = zeros(1,nchk);
for q = 1:nchk
    P = zeros(num_uavs,2);
    for i = 1:num_uavs, P(i,:) = x_true_state_global{i}(1:2, ks(q)).'; end
    r = check_rigidity(P, cfg.R_comm);
    deg_min(q)  = min(r.degrees);  deg_mean(q) = mean(r.degrees);
    globrig(q)  = isequal(r.glob_rigid, true);
    tpe(q)      = r.tri_per_edge;
end
cfg.diag = struct('t', tvec_sim(ks), 'deg_min', deg_min, 'deg_mean', deg_mean, ...
                  'glob_rigid', globrig, 'tri_per_edge', tpe);

fprintf('\n--- generate_swarm_trajectory ---\n');
fprintf('  formation        %s, %d drones, spacing %.0f m\n', ...
        num2str(cfg.formation), num_uavs, cfg.spacing);
fprintf('  path             %s, %.0f m over %.0f s (mean speed %.1f m/s)\n', ...
        cfg.path, cfg.path_length, cfg.T, cfg.mean_speed);
fprintf('  theta convention %s\n', cfg.theta_convention);
fprintf('  ranging radius   %.0f m  ->  R/spacing = %.2f\n', cfg.R_comm, cfg.R_comm/cfg.spacing);
fprintf('  degree           min %d, mean %.1f\n', min(deg_min), mean(deg_mean));
fprintf('  globally rigid   %.0f %% of the time\n', 100*mean(globrig));
fprintf('  triangles/edge   %.2f\n', mean(tpe));
if mean(globrig) < 1
    warning('generate_swarm_trajectory:notAlwaysRigid', ...
            ['The network is globally rigid only %.0f %% of the time. Below ' ...
             '100 %% the formation has discrete ambiguities at some epochs; ' ...
             'either raise R_comm, tighten the spacing, or report this ' ...
             'deliberately as the time-varying-topology case.'], 100*mean(globrig));
end

% ======================= plots ==========================================
if cfg.plots
    figure('Name','swarm truth','Position',[80 80 1180 460]);
    subplot(1,3,1); hold on; grid on; axis equal
    for i = 1:num_uavs
        plot(x_true_state_global{i}(1,:), x_true_state_global{i}(2,:), 'LineWidth', 1);
    end
    plot(c_pos(1,:), c_pos(2,:), 'k--', 'LineWidth', 1.2);
    sel = round(linspace(1, steps_sim, 9));
    for q = sel
        P = zeros(num_uavs,2);
        for i = 1:num_uavs, P(i,:) = x_true_state_global{i}(1:2,q).'; end
        r = check_rigidity(P, cfg.R_comm);
        for e = 1:size(r.edges,1)
            plot(P(r.edges(e,:),1), P(r.edges(e,:),2), '-', 'Color', [.7 .78 .85]);
        end
        plot(P(:,1), P(:,2), 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', [.15 .4 .7], ...
             'MarkerEdgeColor','none');
    end
    title(sprintf('%s path, %s formation', cfg.path, num2str(cfg.formation)));
    xlabel('x [m]'); ylabel('y [m]');

    subplot(1,3,2); hold on; grid on
    plot(tvec_sim, vecnorm(c_vel,2,1), 'LineWidth', 1.2);
    plot(tvec_sim, rad2deg(unwrap(psi))/10, 'LineWidth', 1);
    legend('centroid speed [m/s]','heading [deg/10]','Location','best');
    xlabel('t [s]'); title('speed and heading');

    subplot(1,3,3); hold on; grid on
    plot(cfg.diag.t, cfg.diag.deg_min, 'LineWidth', 1.3);
    plot(cfg.diag.t, cfg.diag.deg_mean, 'LineWidth', 1.3);
    plot(cfg.diag.t, 3*ones(size(cfg.diag.t)), 'k:', 'LineWidth', 1);
    area(cfg.diag.t, 1.5*double(~cfg.diag.glob_rigid), 'FaceAlpha', .15, ...
         'EdgeColor','none','FaceColor',[.8 .3 .35]);
    legend('min degree','mean degree','rigidity floor (3)','NOT globally rigid', ...
           'Location','best');
    xlabel('t [s]'); title('network over time'); ylim([0 max(6, max(deg_mean)+1)]);
end
end
