function [gps_on] = generate_gps_availability(num_drones, K, a, L_out)
% k = 0; good state error
% h = 0; bad state error

r = 1/L_out;
p = (1 - a)/(a * L_out);

gps_on = zeros(K, num_drones);

for i = 1:1:num_drones
    s = (rand < a);                      % start: has a fix
    for k = 1:K
        gps_on(k, i) = s;             % record the current situation
    
        if s
            s = ~(rand < p);       % I have a fix -> lose it with probability p
        else
            s =  (rand < r);       % I have none  -> regain it with probability r
        end
    end
end