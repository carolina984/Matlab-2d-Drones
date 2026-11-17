function alg = alg_spawn_lin()
%ALG_SPAWN_LIN  Gaussian belief, cooperative (linearised ranges).
alg         = alg_pred_lin();                    % same init / predict / estimate
alg.name    = 'SPAWN-lin';
alg.correct = @correct;
end

function S = correct(S, meas, k, cfg)
n = numel(S);
z = meas.z_range(:,:,k);

old_mu = cell(1,n);  old_P = cell(1,n);
for i = 1:n, old_mu{i} = S{i}.mu;  old_P{i} = S{i}.P; end

for f = 1:cfg.num_spawn_iter
    new_mu = cell(1,n);  new_P = cell(1,n);
    for i = 1:n
        mu = S{i}.mu;  
        P = S{i}.P;          % start from the prediction
        for j = 1:n
            if j == i || isnan(z(i,j)), continue; end
            [mu, P] = intermeasurement_range_linearized(old_mu{j}, old_P{j}, mu, P, cfg.range_std, z(i,j));
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
