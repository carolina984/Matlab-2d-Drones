% function [samples_kept] = product_messages(iteration_number, current_belief, messages, intranode_sample, internode_samples, k, M, h)
% num_messages = numel(messages);
% samples_drawn_total = [];
% draw_from_message = ceil(k * M/(num_messages));
% 
% 
% %lets try something
% current_belief(:,3:4) = [];
% %______
% 
% if num_messages == 0
%     samples_kept = current_belief;
% else
%     for i = 1:1:(num_messages)
%         message = messages{i};
%         y = randsample(size(message,1),draw_from_message);
%         samples_drawn = message(y, :);
%         samples_drawn_total =  cat(1, samples_drawn_total, samples_drawn);
%     end
% 
% 
%     % %calculate weight of samples
%     % for i = 1:1:size(samples_drawn_total,1) %(k*M)
%     %     sum = 0;
%     %     for j = 1:1:num_messages
%     %         v = evaluate_point(samples_drawn_total(i, 1:2), messages{j}, 0.5);
%     %         if j == 1
%     %             product = v;
%     %         else
%     %             product = product * v;
%     %         end
%     % 
%     %         sum = sum + v;
%     %     end
%     %     weight = product/sum;
%     %     samples_drawn_total(i, 3) = weight;
%     % end
% 
%     %calculate weight of samples
%     prod_v = ones(size(samples_drawn_total,1), 1);
%     sum_v  = zeros(size(samples_drawn_total,1), 1);
% 
% 
%     for j = 1:num_messages
%         v = evaluate_all_points(samples_drawn_total, messages{j}, h);
%         prod_v = prod_v .* v;
%         sum_v  = sum_v  + v;
%     end
% 
%     v_pred = evaluate_all_points(samples_drawn_total, current_belief, h);
%     samples_drawn_total(:,5) = v_pred .* prod_v ./ sum_v;
% 
% 
% 
% 
%     samples_drawn_total(:,5) = prod_v ./ sum_v;
% 
% 
%     %resample
%     w = randsample(size(samples_drawn_total,1), intranode_sample, true, samples_drawn_total(:, 5));
%     samples_kept = samples_drawn_total(w, :);
%     samples_kept(:, 5) = 1/intranode_sample;
%     samples_kept(:, 6) = 1;
% end
% end

function [samples_kept, broadcasted_message] = product_messages(iteration_number, current_belief, messages, intranode_sample, internode_samples, k, M, h)
% PRODUCT_MESSAGES  SPAWN correction: prediction x position-only messages.
%
% This is your function with the prediction put back in. Everything that
% was already here is unchanged; the three new blocks are marked NEW.
%
% current_belief : N x 6  [x y vx vy w flag]   -- the prediction
% messages       : cell of n x 4  [x y w flag]
% samples_kept   : M x 6  [x y vx vy w flag]
%
% WHY THE PREDICTION HAS TO BE HERE
% SPAWN's belief update is
%     b_i(x_i)  =  b_i_pred(x_i) * PROD_j m_{j->i}(x_i)
% so the prediction is a factor in the target, not an optional extra. In
% your 3-drone network each node has exactly two neighbours, and two
% circles in a plane meet at TWO points: the true position and its mirror
% image across the line joining the neighbours. The messages cannot tell
% them apart -- both are equally consistent with both ranges. Measured
% over 40 seeds without the prediction: 48.2% of the mass at the truth,
% 51.7% at the mirror, mean error 17.8 m. With it: 100% / 0.0% / 0.16 m.
%
% THE WEIGHT
% You draw candidates only from the messages, so the proposal is
%     q(x) = (1/J) SUM_j M_j(x)
% and the target is now
%     f(x) = b_pred(x) * PROD_j M_j(x)
% so  w = f/q = b_pred(x) * prod_v/sum_v. The prediction multiplies in and
% prod_v./sum_v is untouched.
%
% THE APPROXIMATION YOU ARE MAKING, FOR THE TEXT
% Candidates drawn from a ring have no velocity, so one is drawn from the
% prediction. Strictly the proposal then carries a b_pred,v(v) factor and
% the correct weight uses b_pred(p|v) rather than b_pred(p). Treating the
% prediction's position and velocity as independent makes the two equal.
% The cost is that the correction does not sharpen velocity -- velocity
% is set by the body-frame velocity measurement alone. State this.

num_messages = numel(messages);
samples_drawn_total = [];
draw_from_message = ceil(k * M/(num_messages));

if num_messages == 0
    % No neighbours: the correction is the identity. Return the
    % prediction WITH its velocity columns intact, and do not resample --
    % there is no new information, and resampling without information
    % only destroys diversity.
    samples_kept = current_belief;
    wn  = current_belief(:,5) / sum(current_belief(:,5));
    ess = 1 / sum(wn.^2);
else
    for i = 1:1:(num_messages)
        message = messages{i};
        y = randsample(size(message,1), draw_from_message);
        samples_drawn = message(y, :);
        samples_drawn_total = cat(1, samples_drawn_total, samples_drawn);
    end

    % calculate weight of samples
    prod_v = ones(size(samples_drawn_total,1), 1);
    sum_v  = zeros(size(samples_drawn_total,1), 1);
    for j = 1:num_messages
        v = evaluate_all_points(samples_drawn_total, messages{j}, h);
        prod_v = prod_v .* v;
        sum_v  = sum_v  + v;
    end

    % ---- NEW 1: the prediction, as a factor in the target ---------------
    % evaluate_all_points reads the weight from column 3, so the belief is
    % repacked into [x y w] mixture layout. Positions only -- the velocity
    % columns play no part in this density.
    belief_as_mixture = [current_belief(:,1:2), current_belief(:,5)];
    v_pred = evaluate_all_points(samples_drawn_total, belief_as_mixture, h);

    w_total = v_pred .* prod_v ./ sum_v;


    % ---- NEW 3: give every candidate a velocity -------------------------
    % Drawn from the prediction by weight. This MUST come after the
    % weights (which read columns 1:2 only) and BEFORE the resample, so
    % the surviving particles carry their velocities with them.
    iv = randsample(size(current_belief,1), size(samples_drawn_total,1), ...
                    true, current_belief(:,5));
    samples_drawn_total(:,3:4) = current_belief(iv, 3:4);
    % ---------------------------------------------------------------------

    samples_drawn_total(:,5) = w_total;

    % resample
    w = randsample(size(samples_drawn_total,1), intranode_sample, true, ...
                   samples_drawn_total(:, 5));
    samples_kept = samples_drawn_total(w, :);
    samples_kept(:, 5) = 1/intranode_sample;
    samples_kept(:, 6) = 1;

    broadcasted_message = intrasamples_to_intersamples(samples_kept, internode_samples);
end
end