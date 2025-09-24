X  = (-3:0.01:3);
Y  = sinc(X) + 0.1*randn(length(X),1);
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest  = X(2:2:end);
Ytest  = Y(2:2:end);

% Range of hyperparameters
type = 'function estimation';
gam_values  = [10, 1e3, 1e6];
sig2_values = [0.01, 1, 100];
mse_matrix   = zeros(numel(gam_values), numel(sig2_values));

%%
% Prepare figure
figure;
for i = 1:length(gam_values)
    for j = 1:length(sig2_values)
        gam  = gam_values(i);
        sig2 = sig2_values(j);

        % Train LS-SVM model with RBF kernel, preprocessing on
task = {Xtrain, Ytrain, type, gam, sig2, 'RBF_kernel', 'preprocess'};
        [alpha, b] = trainlssvm(task);

        % Simulate (predict) on test data
        Ypred = simlssvm(task, {alpha, b}, Xtest);

        % Compute MSE
        mse_matrix(i,j) = mean((Ypred - Ytest).^2);

        % Plot true vs estimated outputs
        subplot(length(gam_values), length(sig2_values), (i-1)*length(sig2_values) + j);
        plot(Xtest, Ytest, 'k.', 'MarkerSize', 8); hold on;
        plot(Xtest, Ypred, 'b-', 'LineWidth', 1.2);
        title(sprintf('gam=%.0e, sig2=%.2g (MSE=%.4f)', gam, sig2, mse_matrix(i,j)));
        xlabel('x'); ylabel('y'); grid on; hold off;
    end
end
%%
disp('MSE for each (gam, sig2):');
T = array2table(mse_matrix, 'RowNames', cellstr(num2str(gam_values')), 'VariableNames', strcat('s2_', strrep(string(sig2_values),'.','p')));
disp(T);

%%

type = 'function estimation';
gam_values  = [10, 1e3, 1e6];
sig2_values = [0.01, 1, 100];
kernel_type = 'RBF_kernel';

%%

disp('Tuning with SIMPLEX...');
tic;
[gam_simplex, sig2_simplex, cost_simplex] = tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type, 'preprocess'}, ...
    'simplex', ...                         % optimization method
    'crossvalidatelssvm', ...             % cost function
    {10, 'mse'});                         % 10-fold CV, mean squared error
time_simplex = toc;

fprintf('Simplex tuning results:\n');
fprintf('  gamma = %.4g\n',    gam_simplex);
fprintf('  sigma^2 = %.4g\n',  sig2_simplex);
fprintf('  CV MSE = %.4f\n',     cost_simplex);
fprintf('  time = %.2f seconds\n\n', time_simplex);


%%

global GRID_GAMMA GRID_SIGMA2;
GRID_GAMMA = gam_values;
GRID_SIGMA2 = sig2_values;

%%
disp('Tuning with GRIDSEARCH...');
tic;
[gam_grid, sig2_grid, cost_grid] = tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type, 'preprocess'}, ...
    'gridsearch', 'crossvalidatelssvm', {10, 'mse'});             
time_grid = toc;

fprintf('Gridsearch tuning results:\n');
fprintf('  gamma = %.4g\n',    gam_grid);
fprintf('  sigma^2 = %.4g\n',  sig2_grid);
fprintf('  CV MSE = %.4f\n',     cost_grid);
fprintf('  time = %.2f seconds\n\n', time_grid);

%%

num_runs = 5; 

% Initialize tables for storing results
results_simplex = table('Size', [num_runs, 4], 'VariableTypes', {'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Gamma', 'Sigma2', 'CV_MSE', 'Time_s'});
results_gridsearch = table('Size', [num_runs, 4], 'VariableTypes', {'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Gamma', 'Sigma2', 'CV_MSE', 'Time_s'});

%%

for run = 1:num_runs
    disp(['Simplex Run ', num2str(run), '...']);
    tic;
    [gam_s, sig2_s, cost_s] = ...
        tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type, 'preprocess'}, ...
                 'simplex', 'crossvalidatelssvm', {10, 'mse'});
    time_s = toc;
    results_simplex(run, :) = {gam_s, sig2_s, cost_s, time_s};
end

%%

for run = 1:num_runs
    disp(['Gridsearch Run ', num2str(run), '...']);
    tic;
    [gam_g, sig2_g, cost_g] = ...
        tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type, 'preprocess'}, ...
                 'gridsearch', 'crossvalidatelssvm', {10, 'mse'});
    time_g = toc;
    results_gridsearch(run, :) = {gam_g, sig2_g, cost_g, time_g};
end

%%
disp('=== Simplex Results ===');
disp(results_simplex);

disp('=== Gridsearch Results ===');
disp(results_gridsearch);

% Combine and Display
all_results = [array2table(repmat({'Simplex'}, num_runs, 1), 'VariableNames', {'Method'}), results_simplex;
               array2table(repmat({'Gridsearch'}, num_runs, 1), 'VariableNames', {'Method'}), results_gridsearch];

disp('=== All Results ===');
disp(all_results);

%%

% Bayesian framework
sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;

[~ , alpha , b ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
[~ , gam ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
[~ , sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;

sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 } , 'figure') ;



%%
disp(['Optimized Gamma: ', num2str(gam)]);
disp(['Optimized Sigma²: ', num2str(sig2)]);

gam_opt = gam
sig2_opt = sig2
%%
X = 6.* rand (100 , 3) - 3;
Y = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;


%%
% Feature Importance Data
importance = [3, 2, 1]; % Feature 1 is most important, Feature 3 is least important
features = {'Feature 1', 'Feature 2', 'Feature 3'};

% Create bar plot
figure;
barh(importance, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'yticklabel', features);


% Adjust x-axis labels
xticks([1, 3]);
xticklabels({'Least Important', 'Most Important'});
xlabel('Importance');
title('Feature Importance with Categories');


%%

% Feature subsets
feature_sets = {[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]};
errors = zeros(length(feature_sets), 1);
optimal_params = cell(length(feature_sets), 1);

% Cross-validation with tunelssvm
for i = 1:length(feature_sets)
    subset = feature_sets{i};
    X_subset = X(:, subset);

    % Directly use the subset as selected features
    selected_features = subset;
        % Cross-validate the model with the current feature subset
    errors(i) = crossvalidate({X(:, selected_features), Y, 'f', gam_opt, sig2_opt}, 10, 'mse');

    % Store the selected features
    % Store selected features
    optimal_params{i} = selected_features;
end

% Display results
disp('Cross-validation MSE and Selected Features for each feature subset:');
for i = 1:length(feature_sets)
    disp(['Selected Features: ', mat2str(optimal_params{i}), ' - MSE: ', num2str(errors(i))]);
end

%%

%no attention to outliers
X = ( -6:0.2:6)';
Y = sinc ( X ) + 0.1.* rand ( size ( X ) );
out = [15 17 19];
Y ( out ) = 0.7+0.3* rand ( size ( out ) ) ;
out = [41 44 46];
Y ( out ) = 1.5+0.2* rand ( size ( out ) ) ;

model = initlssvm (X , Y , 'f', [] , [] , 'RBF_kernel') ;
costFun = 'crossvalidatelssvm';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mse'}) ;
plotlssvm ( model ) ;

%%

%robust estimation

model = initlssvm (X , Y , 'f', [] , [] , 'RBF_kernel') ;
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm ( model , 'simplex', costFun , {10 , 'mae'} , wFun ) ;
model = robustlssvm ( model ) ;
plotlssvm ( model ) ;

%%

%different wFun

% Define the weighting functions to iterate through
weighting_functions = {'whampel', 'wlogistic', 'wmyriad'};

% Loop over the weighting functions
for k = 1:length(weighting_functions)
    % Select the current weighting function
    wFun = weighting_functions{k};
    
    % Initialize the model
    model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
    
    % Tuning the model using robust cross-validation with the specific weight function
    costFun = 'rcrossvalidatelssvm';
    model = tunelssvm(model, 'simplex', costFun, {10, 'mae'}, wFun);
    
    % Apply robust LS-SVM estimation using the tuned parameters
    model = robustlssvm(model);
    
    % Plot the resulting model
    figure;
    plotlssvm(model);
    title(['LS-SVM with ', wFun, ' weighting function']);
end
