function cfg = sim_config(ovr)
%SIM_CONFIG  Every parameter in one place. Pass a struct to override any field.
%
%   cfg = sim_config()
%   cfg = sim_config(struct('sigma_theta', 0.05, 'num_spawn_iter', 8))

cfg = struct();


%general
cfg.num_uavs = 7;

% ---- trajectory --------------------------------------------------------
cfg.traj = struct('formation_spacing', 30, 'speed', 7, ...
                  'lane_length', 400, 'lane_spacing', 200, 'n_lanes', 5, ...
                  'dt', 0.1, 'sweep', -1);
% ---- timing ------------------------------------------------------------
cfg.dt     = 0.5;              % filter step [s]
cfg.dt_sim = cfg.traj.dt;      % trajectory sample step [s]
cfg.k_last = 800;

% ---- sensors -----------------------------------------------------------
cfg.sigma_theta  = 0.15;       % heading sd [rad]
cfg.sigma_a      = 0.2;        % accelerometer sd [m/s^2]
cfg.sigma_process = 0.5;           % motion model
cfg.velocity_std = 0.3;        % body-frame velocity sd [m/s]
cfg.range_std    = 0.3;        % ranging sd [m]
cfg.lidar_std   = 0.1;
cfg.gps_std      = 0.5;        % GPS sd [m]

% ---- GPS availability (simplified Gilbert, two-state Markov) ----------
cfg.gps_availability = 0.05;   % long-run fraction of steps with a fix
cfg.gps_mean_outage  = 20;     % mean outage length [steps]

% who gets gps

%cfg.gps_receivers = 2;        % one anchor
%cfg.gps_receivers = [2 5];    % two, on opposite sides of the hex
cfg.gps_receivers = 1:7;      % everyone (the line becomes a no-op)
% cfg.gps_receivers = [];       % nobody — the fully GPS-denied baseline

% ---- cooperation -------------------------------------------------------
cfg.range_max          = 32;
cfg.correction_t = 2;  % [s] between SPAWN rounds
cfg.num_spawn_iter       = 4;

% ---- particles ---------------------------------------------------------
cfg.intra_samples = 2000;
cfg.inter_samples = 500;
cfg.kk            = 2;         % candidate oversampling
cfg.h             = 0.3;       % kernel bandwidth

% ---- initialisation ----------------------------------------------------
cfg.start = 1;

if cfg.start == 0
    %Warm Start
    cfg.p_pos_0 = 1.0;             
    cfg.p_vel_0 = 0.1;
else 
    if cfg.start == 1
        %Cold Start
        cfg.p_pos_0 = 5.0;             
        cfg.p_vel_0 = 1.0;
    end
end

% ---- apply overrides ---------------------------------------------------
if nargin >= 1 && ~isempty(ovr)
    for f = fieldnames(ovr).'
        cfg.(f{1}) = ovr.(f{1});
    end
end

% ---- derived (after overrides, so a sweep stays consistent) -----------
cfg.A_noise = cfg.sigma_a^2      * eye(2);
cfg.V_cov   = cfg.velocity_std^2 * eye(2);
cfg.gps_cov = cfg.gps_std^2      * eye(2);
cfg.Q_cov   = cfg.sigma_process^2 * eye(4);
cfg.P_0     = diag([cfg.p_pos_0^2, cfg.p_pos_0^2, cfg.p_vel_0^2, cfg.p_vel_0^2]);

cfg.A = [1 0 cfg.dt 0;
     0 1 0 cfg.dt;
     0 0 1 0;
     0 0 0 1];

cfg.B = [0.5*cfg.dt^2 0;
     0 0.5*cfg.dt^2;
     cfg.dt 0;
     0 cfg.dt];

cfg.H_vel = [0 0 1 0;
         0 0 0 1];
end
