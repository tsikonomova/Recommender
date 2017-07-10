%% Cost function and Gradient descent
%  

function [J, grad] = costFunc(params, Y, R, num_users, num_classes, ...
                                  num_features, lambda)

X = reshape(params(1:num_classes*num_features), num_classes, num_features);
Theta = reshape(params(num_classes*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J = (1/2)*sum(sum((R.*(X*Theta' - Y)).^2));
                       
J = J'+((lambda/2)*sum(sum(Theta.^2)))+((lambda/2)*sum(sum(X.^2)));
                       
X_grad = ((R.*(X*Theta' - Y))*Theta)+(lambda*X);
Theta_grad = ((R.*(X*Theta' - Y))'*X)+(lambda*Theta);

grad = [X_grad(:); Theta_grad(:)];

end
