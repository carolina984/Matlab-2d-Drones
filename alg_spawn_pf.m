function alg = alg_spawn_pf()
%ALG_SPAWN_PF  Particle belief, cooperative (SPAWN / NBP).
alg         = alg_pf();                     % same init / predict / estimate
alg.name    = 'SPAWN-PF';
alg.correct = @correct;
end

function S = correct(S, meas, k, cfg)
n = numel(S);
z = meas.z_range(:,:,k);

pred = S;                                   % the prediction, kept for every round

for f = 1:cfg.num_spawn_iter
    old = S;                                % broadcasts frozen for this round
    new = S;
    for i = 1:n
        msgs = {};
        for j = 1:n
            if j == i || isnan(z(i,j)), continue; end
            if old{j}.bel(1,6) == 0,    continue; end     % j has no information
            msgs{end+1} = compute_message(old{j}.bcast, cfg.inter_samples, ...
                cfg.intra_samples, z(i,j), cfg.range_std);      %#ok<AGROW>
        end
        [new{i}.bel, new{i}.bcast] = product_messages(f, pred{i}.bel, msgs, ...
            cfg.intra_samples, cfg.inter_samples, cfg.kk, cfg.intra_samples, cfg.h);
    end
    S = new;                                % everyone updates together
end
end
