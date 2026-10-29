function [x_true_state_global, true_velocity_local, true_acc_global, ...
          true_acc_local, true_theta, sim_data, info] = ...
          generate_lawnmower_trajectory(cfg)
%GENERATE_LAWNMOWER_TRAJECTORY  Smooth boustrophedon survey path for a hex-7 swarm.
%
%   [x_true_state_global, true_velocity_local, true_acc_global, ...
%    true_acc_local, true_theta, sim_data] = generate_lawnmower_trajectory(cfg)
%
%   OUTPUTS (N samples, one cell per drone; last row of each is time)
%     x_true_state_global{i}  5 x N   [x; y; vx; vy; t]   global frame
%     true_velocity_local{i}  3 x N   [vx_body; vy_body; t]
%     true_acc_global{i}      3 x N   [ax; ay; t]
%     true_acc_local{i}       3 x N   [ax_body; ay_body; t]
%     true_theta{i}           1 x N   heading [rad], unwrapped
%     sim_data                [T; dt_sim; num_uavs]   T = duration [s]
%     info                    diagnostics struct (optional 7th output)
%
%   NOTE ON sim_data(1): T is the DURATION in seconds, so the sample count is
%   N = round(T/dt_sim) + 1, also given as info.N. If your main script treats
%   sim_data(1) as a step count instead, change the one line marked
%   "sim_data" below to  sim_data = [info.N; cfg.dt; n];
%
%   CONFIG, with defaults:
%     cfg.formation_spacing = 70     hex side = circumradius [m]
%     cfg.speed             = 10     constant ground speed of the centroid [m/s]
%     cfg.lane_length       = 400    straight run [m]
%     cfg.lane_spacing      = 100    distance between lanes [m]
%     cfg.n_lanes           = 5      number of straight runs
%     cfg.dt                = 0.1    sample interval [s]
%     cfg.rotate_formation  = false  true = rigid body yaws with the path
%     cfg.theta_mode        = 'course'   'course' | 'body'  (see below)
%     cfg.turn_smoothing    = 0.3    clothoid fraction of each turn, [0,1)
%     cfg.num_uavs          = 7      7 = hex (centre + ring); else a ring
%     cfg.origin            = [0 0]  start of lane 1 [m]
%     cfg.plot              = false
%
% -------------------------------------------------------------------------
% WHY THIS IS BUILT FROM A CURVATURE PROFILE
%
% The obvious lawnmower lays down straight lanes and joins them with
% semicircles. That path is C1 but not C2: curvature jumps from 0 to 1/R at
% every lane end, so centripetal acceleration steps discontinuously and, if
% the formation yaws, angular acceleration is an impulse -- which
% differentiated numerically becomes a spike of size v^2/(R*dt), hundreds of
% m/s^2 on a long formation arm. That is the class of artifact that produced
% the 1367 m error before, and it would go straight into the accelerometer.
%
% So the path is defined by its curvature along arc length, kappa(s), with
% linear ramps between 0 and +-1/R. A linear curvature ramp is a clothoid
% (Euler spiral), which is what road and rail alignment uses for exactly this
% reason. Then, with constant speed v:
%
%     psi(s) = psi0 + integral kappa ds          (C1 heading)
%     c(s)   = c0  + integral [cos psi, sin psi] ds
%     cdot   = v   * [cos psi, sin psi]
%     cddot  = v^2 * kappa * [-sin psi, cos psi]
%     psidot = v   * kappa                       (continuous)
%     psiddot= v^2 * dkappa/ds                   (bounded, piecewise const)
%
% Every derivative is analytic given kappa, so nothing is finite-differenced
% and there are no spikes to clean up. clean_trajectory should find nothing to
% repair here; if it does, that is a bug worth chasing.
%
% Turn length is set so the heading change is exactly pi, and the radius is
% solved by bisection so the net lateral offset equals cfg.lane_spacing
% exactly. With clothoid ramps a 180 degree turn offsets by MORE than 2R, so
% R < lane_spacing/2  (R = 0.482*lane_spacing at smoothing 0.3).
%
% THETA_MODE -- READ THIS IF YOU ROTATE THE FORMATION
% 'body'   theta = psi, the formation's attitude, shared by all drones.
%          Correct for a rigid yawing body, but an off-centre drone then has
%          LATERAL velocity in its own frame: v_local = [v(1-kappa*py);
%          v*kappa*px], up to 22.5 degrees of sideslip at arm 40 m / lane
%          spacing 200 m. A prediction model of the form
%          x += v*dt*[cos th; sin th] cannot represent that component and
%          loses exactly  arm*dpsi = pi*arm  of position per turn -- 126 m at
%          arm 40, 220 m at arm 70 -- INDEPENDENT of speed and lane spacing,
%          so it cannot be tuned away.
% 'course' theta = the direction each drone is actually moving, so
%          v_local = [|v_i|; 0] and sideslip is zero by construction. This is
%          what a non-holonomic prediction model expects, and physically it is
%          a drone that points where it flies. DEFAULT. Positions stay rigid
%          (all 12 edges constant, checked below); only the attitudes differ.
%          In translating mode 'course' and 'body' are identical.
%
% ROTATING VS TRANSLATING
% A rigid formation yawing with the path forces outer drones onto a wider arc
% at the centroid's angular rate: |v_i|/v = sqrt((1-kappa*py)^2+(kappa*px)^2).
% With a 70 m arm that is 2.4x the centroid speed at 100 m lane spacing and
% two drones fly BACKWARD. Translating keeps every drone at the centroid
% velocity with exactly constant geometry. Rotation is refused when the speed
% ratio exceeds cfg.max_speed_ratio or any drone reverses.
% -------------------------------------------------------------------------

if nargin < 1, cfg = struct(); end
D = struct('formation_spacing',70, 'speed',10, 'lane_length',400, ...
           'lane_spacing',100, 'n_lanes',5, 'dt',0.1, ...
           'rotate_formation',false, 'turn_smoothing',0.3, ...
           'num_uavs',7, 'origin',[0 0], 'plot',false, ...
           'max_speed_ratio',2.0, 'ds',0.01, 'theta_mode','course');
fn = fieldnames(D);
for a = 1:numel(fn)
    if ~isfield(cfg, fn{a}) || isempty(cfg.(fn{a})), cfg.(fn{a}) = D.(fn{a}); end
end
assert(cfg.turn_smoothing >= 0 && cfg.turn_smoothing < 1, ...
       'turn_smoothing must be in [0,1).');
assert(cfg.n_lanes >= 1, 'n_lanes must be at least 1.');

v = cfg.speed;  ds = cfg.ds;

%% ---- 1. formation offsets in the body frame ----------------------------
% Drone 1 at the centre, the rest on a ring at radius = spacing, so every
% ring-to-ring and ring-to-centre edge is exactly cfg.formation_spacing
% (hexagon circumradius = side). That is the 12-edge, globally rigid hex-7.
R_f = cfg.formation_spacing;
if cfg.num_uavs == 7
    ang = (0:5) * pi/3;
    P   = [ [0;0], R_f*[cos(ang); sin(ang)] ];        % 2 x 7
else
    ang = (0:cfg.num_uavs-1) * 2*pi/cfg.num_uavs;
    P   = R_f*[cos(ang); sin(ang)];
    warning('generate_lawnmower_trajectory:ring', ...
        ['num_uavs = %d, so the formation is a plain ring of radius %.1f m, ' ...
         'not the hex-7. Edge lengths are NOT all equal and the rigidity ' ...
         'argument for hex-7 does not carry over.'], cfg.num_uavs, R_f);
end
n   = size(P,2);
arm = max(vecnorm(P, 2, 1));

%% ---- 2. solve the turn radius for the requested lateral offset ---------
R_turn = solve_turn_radius(cfg.lane_spacing, cfg.turn_smoothing, ds);

% Speed on a turn for a body yawing at omega = v*kappa. A drone at body
% offset (px,py) has v_i = v(1-kappa*py) xhat + v*kappa*px yhat, so
%   |v_i|/v = sqrt((1-kappa*py)^2 + (kappa*px)^2)
% and (1-kappa*py) < 0 means that drone flies BACKWARD. The loose bound
% 1+arm/R overestimates (2.451 vs an actual 2.371 for hex-7 at the
% defaults), so use the exact per-drone value.
kap_max = 1/R_turn;
sfac    = sqrt((1 - kap_max*P(2,:)).^2 + (kap_max*P(1,:)).^2);
alpha   = R_turn / cfg.lane_spacing;      % R scales linearly with lane spacing
if cfg.rotate_formation
    if max(sfac) > cfg.max_speed_ratio || any(1 - kap_max*P(2,:) < 0)
        R_need = arm/(cfg.max_speed_ratio - 1);
        error('generate_lawnmower_trajectory:tooTight', ...
          ['A rigid %.1f m formation cannot yaw through a %.2f m turn ' ...
           'radius.\n  Drone speeds would span %.3fv to %.3fv (%.1f to ' ...
           '%.1f m/s at v = %.1f), and %d of them fly BACKWARD.\n' ...
           'Fix it by one of:\n' ...
           '  lane_spacing >= %.1f m   (R = %.5f*lane_spacing, so 2*arm is ' ...
           'NOT enough)\n' ...
           '  formation_spacing <= %.1f m\n' ...
           '  rotate_formation = false'], ...
           R_f, R_turn, min(sfac), max(sfac), min(sfac)*v, max(sfac)*v, v, ...
           nnz(1 - kap_max*P(2,:) < 0), R_need/alpha, alpha, ...
           R_turn*(cfg.max_speed_ratio - 1));
    end
end

%% ---- 3. build kappa(s) for the whole path ------------------------------
% Lanes alternate +x / -x. Each turn reverses heading and steps
% +lane_spacing in y, so the turn sense alternates CCW, CW, CCW, ... and psi
% oscillates between 0 and pi rather than winding up.
s_seg = {};  k_seg = {};  dk_seg = {};
for lane = 1:cfg.n_lanes
    [sa, ka, dka] = straight_profile(cfg.lane_length, ds);
    s_seg{end+1} = sa;  k_seg{end+1} = ka;  dk_seg{end+1} = dka;   %#ok<AGROW>
    if lane < cfg.n_lanes
        sgn = (-1)^(lane+1);                    % +1 = CCW
        [sa, ka, dka] = turn_kappa(R_turn, cfg.turn_smoothing, ds);
        s_seg{end+1} = sa;  k_seg{end+1} = sgn*ka;  dk_seg{end+1} = sgn*dka; %#ok<AGROW>
    end
end

s_fine = []; kappa = []; dkappa = []; s0 = 0;
for a = 1:numel(s_seg)
    s_fine = [s_fine, s0 + s_seg{a}];   kappa = [kappa, k_seg{a}];   %#ok<AGROW>
    dkappa = [dkappa, dk_seg{a}];                                    %#ok<AGROW>
    s0 = s0 + s_seg{a}(end) + ds;
end
L_total = s_fine(end);

%% ---- 4. integrate to heading and position ------------------------------
psi_f = cumtrapz(s_fine, kappa);                       % psi0 = 0 (+x)
cx_f  = cfg.origin(1) + cumtrapz(s_fine, cos(psi_f));
cy_f  = cfg.origin(2) + cumtrapz(s_fine, sin(psi_f));

%% ---- 5. sample at constant speed, every dt -----------------------------
N   = floor(L_total/(v*cfg.dt)) + 1;
t   = (0:N-1) * cfg.dt;
s   = min(v * t, L_total);

psi  = interp1(s_fine, psi_f,  s, 'linear', 'extrap');
cx   = interp1(s_fine, cx_f,   s, 'linear', 'extrap');
cy   = interp1(s_fine, cy_f,   s, 'linear', 'extrap');
kap  = interp1(s_fine, kappa,  s, 'linear', 'extrap');
dkap = interp1(s_fine, dkappa, s, 'linear', 'extrap');

c      = [cx; cy];
c_vel  = v   * [cos(psi); sin(psi)];
c_acc  = v^2 * (kap .* [-sin(psi); cos(psi)]);
psidot = v   * kap;
psidd  = v^2 * dkap;

%% ---- 6. per-drone states ----------------------------------------------
x_true_state_global = cell(1,n);
true_velocity_local = cell(1,n);
true_acc_global     = cell(1,n);
true_acc_local      = cell(1,n);
true_theta          = cell(1,n);

Jm = [0 -1; 1 0];                                       % rotate by +90 deg
for i = 1:n
    p = P(:,i);
    if cfg.rotate_formation
        off  = [cos(psi).*p(1) - sin(psi).*p(2);
                sin(psi).*p(1) + cos(psi).*p(2)];
        Jp   = Jm*p;
        offJ = [cos(psi).*Jp(1) - sin(psi).*Jp(2);
                sin(psi).*Jp(1) + cos(psi).*Jp(2)];
        pos  = c + off;
        vel  = c_vel + psidot .* offJ;
        acc  = c_acc + psidd .* offJ - (psidot.^2) .* off;
    else
        pos  = c + p;                % offset fixed in the GLOBAL frame
        vel  = c_vel;
        acc  = c_acc;
    end

    % which frame is "local"?  See THETA_MODE in the header.
    switch lower(cfg.theta_mode)
        case 'body'
            th = psi;
        case 'course'
            if min(vecnorm(vel,2,1)) < 1e-6 * max(v, eps)
                error('generate_lawnmower_trajectory:zeroSpeed', ...
                  ['Drone %d reaches %.3g m/s, so its course is undefined ' ...
                   '(atan2(0,0)) and the heading would jump. Widen the lanes ' ...
                   'or shrink the formation so no drone stalls.'], ...
                   i, min(vecnorm(vel,2,1)));
            end
            th = unwrap(atan2(vel(2,:), vel(1,:)));
        otherwise
            error('generate_lawnmower_trajectory:thetaMode', ...
                  'theta_mode must be ''body'' or ''course'', got ''%s''.', ...
                  cfg.theta_mode);
    end

    vl = [ cos(th).*vel(1,:) + sin(th).*vel(2,:);
          -sin(th).*vel(1,:) + cos(th).*vel(2,:)];
    al = [ cos(th).*acc(1,:) + sin(th).*acc(2,:);
          -sin(th).*acc(1,:) + cos(th).*acc(2,:)];

    x_true_state_global{i} = [pos; vel; t];             % 5 x N
    true_velocity_local{i} = [vl;  t];                  % 3 x N
    true_acc_global{i}     = [acc; t];                  % 3 x N
    true_acc_local{i}      = [al;  t];                  % 3 x N
    true_theta{i}          = th;                        % 1 x N
end

sim_data = [t(end); cfg.dt; n];      % sim_data -- see NOTE in the header

%% ---- 7. audit, so a bad path announces itself -------------------------
spd_hi = cellfun(@(x) max(vecnorm(x(3:4,:),2,1)), x_true_state_global);
spd_lo = cellfun(@(x) min(vecnorm(x(3:4,:),2,1)), x_true_state_global);
amax   = cellfun(@(x) max(vecnorm(x(1:2,:),2,1)), true_acc_local);
slip   = cellfun(@(x) max(abs(x(2,:))),            true_velocity_local);
dpsi   = max(cellfun(@(x) max(abs(diff(x)))/cfg.dt, true_theta));

pairs = [];
for i = 1:n, for j = i+1:n
    pairs = [pairs; vecnorm(x_true_state_global{i}(1:2,:) - ...
                            x_true_state_global{j}(1:2,:), 2, 1)];  %#ok<AGROW>
end, end
drift = max(max(pairs,[],2) - min(pairs,[],2));

info = struct('N',N, 'duration_s',t(end), 'path_length_m',L_total, ...
    'turn_radius_m',R_turn, 'formation_arm_m',arm, ...
    'R_per_lane_spacing',alpha, 'max_drone_speed',max(spd_hi), ...
    'min_drone_speed',min(spd_lo), 'speed_ratio',max(spd_hi)/v, ...
    'max_accel',max(amax), 'centripetal',v^2/R_turn, ...
    'max_lateral_v_local',max(slip), 'max_heading_rate',dpsi, ...
    'edge_length_drift_m',drift, 'formation',P, ...
    'centroid',c, 'psi',psi, 'kappa',kap, 't',t, 'cfg',cfg);

if cfg.rotate_formation, mode = 'yawing with the path'; else, mode = 'translating'; end
fprintf('\n--- lawnmower: %d drones, %d lanes, %.0f x %.0f m ---\n', ...
    n, cfg.n_lanes, cfg.lane_length, cfg.lane_spacing*(cfg.n_lanes-1));
fprintf('  %d samples, %.1f s, path %.1f m at %.1f m/s\n', N, t(end), L_total, v);
fprintf('  turn radius %.2f m (lane spacing %.1f m, R = %.5f*spacing)\n', ...
    R_turn, cfg.lane_spacing, alpha);
fprintf('  formation %s, arm %.1f m, theta_mode ''%s''\n', mode, arm, cfg.theta_mode);
fprintf('  drone speed      %.3f .. %.3f m/s  (%.3f .. %.3fx centroid)\n', ...
    min(spd_lo), max(spd_hi), min(spd_lo)/v, max(spd_hi)/v);
fprintf('  max |accel|      %.4f m/s^2 (centroid centripetal %.4f)\n', ...
    max(amax), v^2/R_turn);
fprintf('  max heading rate %.4f rad/s\n', dpsi);
fprintf('  max lateral v_local %.4e m/s', max(slip));
if max(slip) < 1e-6
    fprintf('  (no sideslip: a non-holonomic prediction model is exact)\n');
else
    fprintf(['\n     WARNING: body-frame lateral velocity is non-zero. A ' ...
             'prediction model of\n     the form x += v*dt*[cos th; sin th] ' ...
             'will lose about pi*arm = %.1f m per turn.\n     Use ' ...
             'theta_mode = ''course'', or propagate the full 2-D local ' ...
             'velocity.\n'], pi*arm);
end
fprintf('  edge-length drift %.3e m  <-- should be ~0; formation is rigid\n', drift);
fprintf('  sim_data = [%.1f s; %.3f s; %d];  N = %d\n\n', t(end), cfg.dt, n, N);

%% ---- 8. optional picture ----------------------------------------------
if cfg.plot
    figure('Color','w');
    subplot(2,2,[1 3]); hold on; axis equal; grid on
    plot(c(1,:), c(2,:), 'k-', 'LineWidth', 1.4);
    for i = 1:n
        X = x_true_state_global{i};
        plot(X(1,:), X(2,:), '-', 'LineWidth', 0.6);
    end
    for a = round(linspace(1, N, 6))
        pts = cell2mat(cellfun(@(x) x(1:2,a), x_true_state_global, ...
                               'UniformOutput', false));
        plot(pts(1,:), pts(2,:), 'ko', 'MarkerSize', 3, 'MarkerFaceColor', [.6 .6 .6]);
        if size(pts,2) == 7
            plot(pts(1,[2:7 2]), pts(2,[2:7 2]), '-', 'Color', [.6 .6 .6]);
        end
    end
    xlabel('x [m]'); ylabel('y [m]'); title('lawnmower, centroid + drones');

    subplot(2,2,2); plot(t, kap, 'LineWidth', 1.2); grid on
    ylabel('\kappa [1/m]'); title('curvature: ramped, never stepped');

    subplot(2,2,4); hold on; grid on
    for i = 1:n
        plot(t, vecnorm(true_acc_local{i}(1:2,:),2,1), 'LineWidth', 0.8);
    end
    xlabel('t [s]'); ylabel('|a| [m/s^2]'); title('acceleration magnitude');
end
end

% =========================================================================
function [s, k, dk] = straight_profile(L, ds)
nn = max(round(L/ds), 1);
s  = (0:nn-1) * ds;
k  = zeros(1,nn);
dk = zeros(1,nn);
end

function [s, k, dk] = turn_kappa(R, frac, ds)
%TURN_KAPPA  kappa and dkappa/ds for a 180 degree turn, unit (CCW) sense.
%   Ramp 0 -> 1/R over Lt, hold over Lh, ramp back over Lt, with
%   (Lt + Lh)/R = pi so the heading change is exactly pi.
Lt = frac * pi * R;
Lh = pi*R - Lt;
n1 = max(round(Lt/ds), 0);
n2 = max(round(Lh/ds), 1);
if n1 == 0
    s = (0:n2-1)*ds;  k = ones(1,n2)/R;  dk = zeros(1,n2);  return
end
s1  = (0:n1-1)*ds;
k1  =  (s1/Lt)/R;        dk1 =  ones(1,n1)/(R*Lt);
k2  =  ones(1,n2)/R;     dk2 =  zeros(1,n2);
k3  =  (1 - s1/Lt)/R;    dk3 = -ones(1,n1)/(R*Lt);
s   = [s1, Lt + (0:n2-1)*ds, Lt+Lh + s1];
k   = [k1, k2, k3];
dk  = [dk1, dk2, dk3];
end

function R = solve_turn_radius(target, frac, ds)
%SOLVE_TURN_RADIUS  bisect R so a 180 degree turn offsets laterally by target.
%   With clothoid ramps the offset exceeds 2R, so R < target/2.
lo = target/8;  hi = target;
for it = 1:100
    mid = 0.5*(lo+hi);
    if lateral_offset(mid, frac, ds) < target, lo = mid; else, hi = mid; end
end
R = 0.5*(lo+hi);
end

function off = lateral_offset(R, frac, ds)
[s, k] = turn_kappa(R, frac, ds);
psi = cumtrapz(s, k);
off = cumtrapz(s, sin(psi));
off = off(end);
end
