%% Recommender system 
%  
%  Reccomends classes for users
%

%  Load data
load ('classes.mat');

%  Get average review
mean(Y(1, R(1, :)));

%  Plot
imagesc(Y);
ylabel('Classes');
xlabel('Users');

%  Reduce size
num_users = 4; num_classes = 5; num_features = 3;
X = X(1:num_classes, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_classes, 1:num_users);
R = R(1:num_classes, 1:num_users);

%  Evaluate cost function
J = costFunc([X(:) ; Theta(:)], Y, R, num_users, num_classes, ...
               num_features, 0);

J = costFunc([X(:) ; Theta(:)], Y, R, num_users, num_classes, ...
               num_features, 1.5);


% load class list
classList = loadClassList();

%  Initialize my reviews
my_reviews = zeros(1682, 1);

my_reviews(1) = 4;
my_reviews(8) = 3;
my_reviews(13) = 2;
my_reviews(17) = 1;
my_reviews(20) = 6;
my_reviews(33) = 5;
my_reviews(35) = 9;
my_reviews(44) = 8;
my_reviews(55) = 10;
my_reviews(69) = 7;

%  Load data
load('classes.mat');

%  Add reviews 
Y = [my_reviews Y];
R = [(my_reviews ~= 0) R];

%  Normalize Reviews
[Ynorm, Ymean] = normalizeReviews(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_classes = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_classes, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(costFunc(t, Ynorm, R, num_users, num_classes, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_classes*num_features), num_classes, num_features);
Theta = reshape(theta(num_classes*num_features+1:end), ...
                num_users, num_features);

p = X * Theta';
my_predictions = p(:,1) + Ymean;

classList = loadClassList();

[r, ix] = sort(my_predictions, 'descend');

for i=1:10
    j = ix(i);
    fprintf(my_predictions(j));
    fprintf('-\n\n');
    fprintf(classList{j});
    fprintf('==============\n\n');
end
