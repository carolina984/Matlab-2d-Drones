function alg = alg_bp_lidar()
%ALG_SPAWN_LIN  Gaussian belief, cooperative (linearised ranges).
alg         = alg_pred_lin();                    % same init / predict / estimate
alg.name    = 'BP-LiDAR';
alg.correct = @correct;
end

function S = correct(S, meas, k, cfg)
n = numel(S);
z = meas.z_lidar(:,:,:,k);
theta_measured = meas.theta(:,k);

old_mu = cell(1,n);  old_P = cell(1,n);
prev_mess_L = cell(n,n);  %recieved lambda
prev_mess_e = cell(n,n);  %recieved eta

for i = 1:n
    old_mu{i} = S{i}.mu;  
    old_P{i} = S{i}.P; 
end

for f = 1:cfg.num_spawn_iter
    new_mu = cell(1,n);  
    new_P = cell(1,n);

    ndL = cell(n,n);  
    nde = cell(n,n);
    for i = 1:n
        mu = S{i}.mu;  
        P = S{i}.P;          % start from the prediction
        
        lambda = inv(P);
        eta = P\mu;

        for j = 1:n
            if j == i || isnan(z(1,i,j)) 
                continue; 
            end

            mu_j = old_mu{j};  
            P_j = old_P{j};

            if ~isempty(prev_mess_L{j,i}) %exclude previous message from i to j
                L = inv(P_j) - prev_mess_L{j,i};
                e = P_j\mu_j   - prev_mess_e{j,i};
                P_j = inv(L);  
                mu_j = L\e;
            end

            L0 = inv(P);  
            e0 = L0*mu;

            [mu, P] = intermeasurements_lidar(theta_measured(i), cfg.sigma_theta, mu_j, P_j, mu, P, cfg.lidar_std, z(:,i,j));
            
            ndL{i,j} = inv(P) - L0;    % contribution from j to i
            nde{i,j} = P\mu   - e0;
        
        end
        new_mu{i} = mu;  
        new_P{i} = P;
    end
    old_mu = new_mu;  
    old_P = new_P;        % everyone updates together

    prev_mess_L = ndL;  
    prev_mess_e = nde;
end

for i = 1:n
    S{i}.mu = old_mu{i};  
    S{i}.P = old_P{i}; 
end

end
