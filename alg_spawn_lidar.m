function alg = alg_spawn_lidar()
%ALG_SPAWN_LIN  Gaussian belief, cooperative (linearised ranges).
alg         = alg_pred_lin();                    % same init / predict / estimate
alg.name    = 'SPAWN-LiDAR';
alg.correct = @correct;
end

function S = correct(S, meas, k, cfg)
n = numel(S);
z = meas.z_lidar(:,:,:,k);
theta_measured = meas.theta(:,k);

old_mu = cell(1,n);  old_P = cell(1,n);
for i = 1:n
    old_mu{i} = S{i}.mu;  
    old_P{i} = S{i}.P; 
end

for f = 1:cfg.num_spawn_iter
    new_mu = cell(1,n);  new_P = cell(1,n);
    for i = 1:n
        mu = S{i}.mu;  
        P = S{i}.P;          % start from the prediction
        for j = 1:n
            if j == i || isnan(z(1,i,j)), continue; end
            [mu, P] = intermeasurements_lidar(theta_measured(i), cfg.sigma_theta, old_mu{j}, old_P{j}, mu, P, cfg.lidar_std, z(:,i,j));
        end
        new_mu{i} = mu;  new_P{i} = P;
    end
    old_mu = new_mu;  
    old_P = new_P;        % everyone updates together
end

for i = 1:n
    S{i}.mu = old_mu{i};  
    S{i}.P = old_P{i}; 
end

end
