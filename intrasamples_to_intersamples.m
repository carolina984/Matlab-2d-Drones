function [broadcasted_message] = intrasamples_to_intersamples(belief, intersamples)
% Alg. 3 line 6 -- broadcast. This is what goes over the air:
% inter_samples samples of the belief.

idx = randsample(size(belief,1), intersamples, true, belief(:,5));
broadcasted_message = belief(idx, :);

end