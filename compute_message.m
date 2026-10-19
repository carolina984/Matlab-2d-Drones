function [message] = compute_message(neighbor_belief_intersamples, internode_samples, intranode_samples, range_measurement, noise)
% Neighbor belief: [x y vx vy weight flag]
% Estrutura da mensagem: [x y weight flag]

ratio_samples  = intranode_samples/internode_samples;
neighbor_belief_intersamples = repmat(neighbor_belief_intersamples,ratio_samples,1);

message = zeros(intranode_samples, 4);
flag = neighbor_belief_intersamples(1, 6);

for i = 1:1:intranode_samples
    theta = rand * 2 * pi;
    message(i, 1:2) = neighbor_belief_intersamples(i, 1:2) + ((range_measurement)) * [sin(theta), cos(theta)]; %noise * rand())
    message(i, 3) = neighbor_belief_intersamples(i, 5); %particle weight
end
message(1, 4) = flag;
end
