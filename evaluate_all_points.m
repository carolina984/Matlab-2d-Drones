function p = evaluate_all_points(q, mixture, h)
    d2 = (q(:,1) - mixture(:,1).').^2 + (q(:,2) - mixture(:,2).').^2;
    p  = exp(-d2/(2*h^2)) * mixture(:,3) / (2*pi*h^2);
end
