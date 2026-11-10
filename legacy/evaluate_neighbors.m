function [neighbors, range_to_agents, range_to_anchors] = evaluate_neighbors(max_detection_radius, number_agents, number_anchors, agents, anchors)

range_to_agents = zeros(number_agents, number_agents);
range_to_anchors = zeros(number_agents, number_anchors);
neighbors = zeros(number_agents, 2);

for i = 1:1:number_agents
    neighbor_agents = 0;
    neighbor_anchors = 0;
    for j = 1:1:number_agents
        distance_ij = norm(agents(i, :) - agents(j, :));
        if distance_ij < max_detection_radius  
            range_to_agents(i,j) = distance_ij;
            neighbor_agents = neighbor_agents + 1;
        end
    end

    for j = 1:1:number_anchors
        distance_ij = norm(agents(i, :) - anchors(j, :));
        if distance_ij < max_detection_radius  
            range_to_anchors(i,j) = distance_ij;
            neighbor_anchors = neighbor_anchors + 1;
        end
    end

    neighbors(i, 1) = neighbor_agents;
    neighbors(i, 2) = neighbor_anchors;
end

end
