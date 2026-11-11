function [x_true_state_global, true_velocity_local, true_acc_local, ...
          true_theta, sim_data, info] = generate_lawnmower_trajectory(cfg)
%GENERATE_LAWNMOWER_TRAJECTORY  Hex-7 formation on a smooth boustrophedon path.
%
%   OUTPUTS (N samples, one cell per drone; last row of each is time)
%     x_true_state_global{i}  5 x N   [x; y; vx; vy; t]   global frame
%     true_velocity_local{i}  3 x N   [vx_body; vy_body; t]
%     true_acc_local{i}       3 x N   [ax_body; ay_body; t]
%     true_theta{i}           1 x N   heading [rad], CLOCKWISE-POSITIVE
%     sim_data                [T; dt_sim; num_uavs]   T = duration [s]
%
%   CONFIG, with defaults:
%     cfg.formation_spacing = 20     hex side = circumradius [m]
%     cfg.speed             = 8      centroid ground speed [m/s]
%     cfg.lane_length       = 400    straight run [m]
%     cfg.lane_spacing      = 200    distance between lanes [m]
%     cfg.n_lanes           = 5
%     cfg.dt                = 0.1    sample interval [s]
%     cfg.sweep             = -1     -1 sweeps toward -y (first turn right)
%     cfg.turn_smoothing    = 0.3    clothoid fraction of each turn, [0,1)
%     cfg.plot              = false
%
%   FIXED CHOICES (no longer options)
%   - The formation is hex-7 and it YAWS with the path.
%   - theta is each drone's COURSE (the direction it actually moves), so the
%     body-frame velocity is [|v|; 0] and there is no sideslip for a
%     non-holonomic prediction model to lose.
%   - theta is reported CLOCKWISE-POSITIVE, matching this codebase, which
%     rotates body -> global with R(-theta) = [cos th, sin th; -sin th, cos th].
%
%   The path is built from its curvature along arc length with clothoid
%   (linear) ramps, so psi is C1 and the centroid is C2: no acceleration
%   steps and no angular-acceleration impulses to leak into the
%   accelerometer. The turn radius is solved so the lateral offset equals
%   lane_spacing exactly (R = 0.482*lane_spacing at smoothing 0.3).

if nargin < 1, cfg = struct(); end
D = struct('formation_spacing',20, 'speed',8, 'lane_length',400, ...
           'lane_spacing',200, 'n_lanes',5, 'dt',0.1, 'sweep',-1, ...
           'turn_smoothing',0.3, 'plot',false, 'ds',0.01, 'max_speed_ratio',2.0);
for f = fieldnames(D).'
    if ~isfield(cfg,f{1}) || isempty(cfg.(f{1})), cfg.(f{1}) = D.(f{1}); end
end
assert(cfg.turn_smoothing >= 0 && cfg.turn_smoothing < 1, 'turn_smoothing in [0,1).');
assert(cfg.sweep == 1 || cfg.sweep == -1, 'sweep must be +1 or -1.');

v = cfg.speed;  ds = cfg.ds;

%% ---- 1. hex-7 offsets in the body frame -------------------------------
% centre + 6 on a ring at radius = spacing, so all 12 edges are equal
ang = (0:5) * pi/3;
P   = [[0;0], cfg.formation_spacing*[cos(ang); sin(ang)]];      % 2 x 7
n   = 7;
arm = cfg.formation_spacing;

%% ---- 2. turn radius, and the yaw feasibility check --------------------
R_turn = solve_turn_radius(cfg.lane_spacing, cfg.turn_smoothing, ds);

% A yawing rigid body puts drone (px,py) at speed
%   |v_i|/v = sqrt((1-kappa*py)^2 + (kappa*px)^2),
% and (1-kappa*py) < 0 means that drone flies BACKWARD.
kap  = 1/R_turn;
sfac = sqrt((1 - kap*P(2,:)).^2 + (kap*P(1,:)).^2);
if max(sfac) > cfg.max_speed_ratio || any(1 - kap*P(2,:) < 0)
    error('generate_lawnmower_trajectory:tooTight', ...
        ['A %.1f m formation cannot yaw through a %.2f m turn radius: drone ' ...
         'speeds would span %.2fv to %.2fv.\nRaise lane_spacing (R = %.4f*' ...
         'lane_spacing) or lower formation_spacing to <= %.1f m.'], ...
        arm, R_turn, min(sfac), max(sfac), R_turn/cfg.lane_spacing, ...
        R_turn*(cfg.max_speed_ratio-1));
end

%% ---- 3. curvature profile along the whole path ------------------------
% Lanes alternate +x/-x. Positive kappa turns left, so the sense alternates
% and psi oscillates between 0 and pi instead of winding up.
s_seg = {}; k_seg = {}; dk_seg = {};
for lane = 1:cfg.n_lanes
    nn = max(round(cfg.lane_length/ds),1);
    s_seg{end+1}=(0:nn-1)*ds; k_seg{end+1}=zeros(1,nn); dk_seg{end+1}=zeros(1,nn); %#ok<AGROW>
    if lane < cfg.n_lanes
        sgn = cfg.sweep * (-1)^(lane+1);
        [sa, ka, dka] = turn_kappa(R_turn, cfg.turn_smoothing, ds);
        s_seg{end+1}=sa; k_seg{end+1}=sgn*ka; dk_seg{end+1}=sgn*dka;   %#ok<AGROW>
    end
end

s_fine=[]; kappa=[]; dkappa=[]; s0=0;
for a = 1:numel(s_seg)
    s_fine=[s_fine, s0+s_seg{a}]; kappa=[kappa, k_seg{a}];             %#ok<AGROW>
    dkappa=[dkappa, dk_seg{a}];                                        %#ok<AGROW>
    s0 = s0 + s_seg{a}(end) + ds;
end
L_total = s_fine(end);

%% ---- 4. integrate to heading and centroid position -------------------
psi_f = cumtrapz(s_fine, kappa);
cx_f  = cumtrapz(s_fine, cos(psi_f));
cy_f  = cumtrapz(s_fine, sin(psi_f));

%% ---- 5. sample at constant speed, every dt ---------------------------
N = floor(L_total/(v*cfg.dt)) + 1;
t = (0:N-1) * cfg.dt;
s = min(v*t, L_total);

psi  = interp1(s_fine, psi_f,  s, 'linear','extrap');
c    = [interp1(s_fine, cx_f, s, 'linear','extrap');
        interp1(s_fine, cy_f, s, 'linear','extrap')];
kp   = interp1(s_fine, kappa,  s, 'linear','extrap');
dkp  = interp1(s_fine, dkappa, s, 'linear','extrap');

c_vel  = v   * [cos(psi); sin(psi)];
c_acc  = v^2 * (kp .* [-sin(psi); cos(psi)]);
psidot = v   * kp;
psidd  = v^2 * dkp;

%% ---- 6. per-drone states ---------------------------------------------
x_true_state_global = cell(1,n);
true_velocity_local = cell(1,n);
true_acc_local      = cell(1,n);
true_theta          = cell(1,n);

Jm = [0 -1; 1 0];
for i = 1:n
    p  = P(:,i);
    Jp = Jm*p;

    off  = [cos(psi)*p(1)  - sin(psi)*p(2);  sin(psi)*p(1)  + cos(psi)*p(2)];
    offJ = [cos(psi)*Jp(1) - sin(psi)*Jp(2); sin(psi)*Jp(1) + cos(psi)*Jp(2)];

    pos = c + off;
    vel = c_vel + psidot .* offJ;
    acc = c_acc + psidd .* offJ - (psidot.^2) .* off;

    % course: the direction this drone actually moves
    th = unwrap(atan2(vel(2,:), vel(1,:)));

    % body frame = global rotated by -th, which makes vl = [|v|; 0]
    vl = [ cos(th).*vel(1,:) + sin(th).*vel(2,:);
          -sin(th).*vel(1,:) + cos(th).*vel(2,:)];
    al = [ cos(th).*acc(1,:) + sin(th).*acc(2,:);
          -sin(th).*acc(1,:) + cos(th).*acc(2,:)];

    x_true_state_global{i} = [pos; vel; t];
    true_velocity_local{i} = [vl;  t];
    true_acc_local{i}      = [al;  t];
    true_theta{i}          = -th;          % clockwise-positive
end

sim_data = [t(end); cfg.dt; n];

%% ---- 7. audit --------------------------------------------------------
spd  = cellfun(@(x) max(vecnorm(x(3:4,:),2,1)), x_true_state_global);
spdl = cellfun(@(x) min(vecnorm(x(3:4,:),2,1)), x_true_state_global);
amax = cellfun(@(x) max(vecnorm(x(1:2,:),2,1)), true_acc_local);
slip = cellfun(@(x) max(abs(x(2,:))),           true_velocity_local);

pairs = [];
for i = 1:n
    for j = i+1:n
    pairs = [pairs; vecnorm(x_true_state_global{i}(1:2,:) - ...
                            x_true_state_global{j}(1:2,:),2,1)];       %#ok<AGROW>
    end
end
drift = max(max(pairs,[],2) - min(pairs,[],2));

info = struct('N',N, 'duration_s',t(end), 'path_length_m',L_total, ...
    'turn_radius_m',R_turn, 'formation_arm_m',arm, ...
    'speed_range',[min(spdl) max(spd)], 'max_accel',max(amax), ...
    'max_lateral_v_local',max(slip), 'edge_length_drift_m',drift, ...
    'formation',P, 'centroid',c, 'psi',psi, 'kappa',kp, 't',t, 'cfg',cfg);

fprintf('\n--- lawnmower: hex-7 arm %.1f m, %d lanes, %.0f x %.0f m ---\n', ...
    arm, cfg.n_lanes, cfg.lane_length, cfg.lane_spacing*(cfg.n_lanes-1));
fprintf('  %d samples, %.1f s, path %.1f m, turn radius %.2f m\n', ...
    N, t(end), L_total, R_turn);
fprintf('  drone speed  %.3f .. %.3f m/s  (%.2f .. %.2fx centroid)\n', ...
    min(spdl), max(spd), min(spdl)/v, max(spd)/v);
fprintf('  max |accel|  %.4f m/s^2,  lateral v_local %.2e m/s\n', max(amax), max(slip));
fprintf('  edge drift   %.2e m  <-- ~0, formation is rigid\n', drift);
fprintf('  set k_last <= %d at dt = 0.5\n\n', floor((N-1)*cfg.dt/0.5)+1);

%% ---- 8. optional picture ---------------------------------------------
if cfg.plot
    figure('Color','w');
    subplot(1,2,1); hold on; axis equal; grid on
    plot(c(1,:), c(2,:), 'k-', 'LineWidth',1.4);
    for i = 1:n
        X = x_true_state_global{i};  plot(X(1,:), X(2,:), '-', 'LineWidth',0.6);
    end
    for a = round(linspace(1,N,6))
        pts = cell2mat(cellfun(@(x) x(1:2,a), x_true_state_global, 'UniformOutput',false));
        plot(pts(1,[2:7 2]), pts(2,[2:7 2]), '-', 'Color',[.6 .6 .6]);
        plot(pts(1,:), pts(2,:), 'ko', 'MarkerSize',3, 'MarkerFaceColor',[.6 .6 .6]);
    end
    xlabel('x [m]'); ylabel('y [m]'); title('path + formation');

    subplot(1,2,2); hold on; grid on
    for i = 1:n, plot(t, vecnorm(true_acc_local{i}(1:2,:),2,1), 'LineWidth',0.8); end
    xlabel('t [s]'); ylabel('|a| body [m/s^2]'); title('acceleration');
end
end

% =========================================================================
function [s, k, dk] = turn_kappa(R, frac, ds)
% kappa and dkappa/ds for a 180 deg turn: ramp up, hold, ramp down, with
% (Lt + Lh)/R = pi so the heading change is exactly pi.
Lt = frac*pi*R;  Lh = pi*R - Lt;
n1 = max(round(Lt/ds),0);  n2 = max(round(Lh/ds),1);
if n1 == 0
    s = (0:n2-1)*ds;  k = ones(1,n2)/R;  dk = zeros(1,n2);  return
end
s1 = (0:n1-1)*ds;
s  = [s1, Lt+(0:n2-1)*ds, Lt+Lh+s1];
k  = [(s1/Lt)/R,  ones(1,n2)/R,       (1-s1/Lt)/R];
dk = [ones(1,n1)/(R*Lt), zeros(1,n2), -ones(1,n1)/(R*Lt)];
end

function R = solve_turn_radius(target, frac, ds)
% bisect R so a 180 deg turn offsets laterally by `target`
lo = target/8;  hi = target;
for it = 1:100
    mid = 0.5*(lo+hi);
    [s,k] = turn_kappa(mid, frac, ds);
    off = cumtrapz(s, sin(cumtrapz(s,k)));
    if off(end) < target, lo = mid; else, hi = mid; end
end
R = 0.5*(lo+hi);
end