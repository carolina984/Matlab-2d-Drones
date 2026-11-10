function [x_evaluated] = evaluate_point(x, mixture, h)
number_samples = size(mixture, 1);
x_evaluated = 0;
for i = 1:1:number_samples
    x_evaluated = x_evaluated + mixture(i, 3) * mvnpdf(x, mixture(i, 1:2), h^2*eye(2));
end

end
