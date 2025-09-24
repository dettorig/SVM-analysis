type = 'c';         % Classification problem
gam = 1;            % Regularization parameter
sigma2_values = [0.01, 0.1, 1, 10, 20];  % Five different sigma² values to test

for i = 1:length(sigma2_values)
    sigma2 = sigma2_values(i);
    
    disp(['RBF kernel with sigma² = ', num2str(sigma2)]);
    
    % Train the LS-SVM model with RBF kernel (Note that LS-SVMlab expects sigma²)
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'});
    
    % Plot the decision boundary using training data; 'preprocess' is used if your data needs scaling
    figure;
    plotlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel', 'preprocess'}, {alpha, b});
    title(['LS-SVM Decision Boundary - RBF Kernel (sigma² = ', num2str(sigma2), ')']);
    
    % Predict on the test set
    [Yht, Zt] = simlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'}, {alpha, b}, Xtest);
    
    % Compute the test error
    err = sum(Yht ~= Ytest);
    fprintf('\nOn test set (sigma² = %.2f): #misclassified = %d, error rate = %.2f%%\n', sigma2, err, err/length(Ytest)*100);
    
    disp('Press any key to continue to the next sigma² value...');
    pause;
end
%% 


type = 'c';          
sigma2 = 1;          
gam_values = [0.02, 10, 1000, 10000, 1000000];  

for i = 1:length(gam_values)
    gam = gam_values(i);
   
    
    disp(['RBF kernel with sigma² = ', num2str(sigma2), ' and gamma = ', num2str(gam)]);
    
    % Train the LS-SVM model
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'});
    
    % Plot the decision boundary
    figure;
    plotlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel', 'preprocess'}, {alpha, b});
    title(['LS-SVM Decision Boundary - RBF Kernel (sigma² = ', num2str(sigma2), ', \gamma = ', num2str(gam), ')']);
    
    % Predict on the test set
    [Yht, Zt] = simlssvm({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'}, {alpha, b}, Xtest);
    
    % Compute the test error
    err = sum(Yht ~= Ytest);
    fprintf('\nOn test set (gamma = %.2f): #misclassified = %d, error rate = %.2f%%\n', gam, err, err/length(Ytest)*100);
    
    disp('Press any key to continue to the next gamma value...');
    pause;
end

%%

type = 'c';   % Classification
gam_values = [0.02, 10, 1000, 10000, 1000000];  
sigma2_values = [0.01, 0.1, 1, 10, 20];  

% Preallocate performance matrices
perf_random = zeros(length(gam_values), length(sigma2_values));
perf_kfold = zeros(length(gam_values), length(sigma2_values));
perf_leaveoneout = zeros(length(gam_values), length(sigma2_values));

for i = 1:length(gam_values)
    gam = gam_values(i);
    for j = 1:length(sigma2_values)
        sigma2 = sigma2_values(j);

        % Random split validation (80% train, 20% validation)
        perf_random(i,j) = rsplitvalidate({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'}, 0.80, 'misclass');

        % 10-fold cross-validation
        perf_kfold(i,j) = crossvalidate({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'}, 10, 'misclass');

        % Leave-one-out cross-validation
        perf_leaveoneout(i,j) = leaveoneout({Xtrain, Ytrain, type, gam, sigma2, 'RBF_kernel'}, 'misclass');
    end
end

% Find best (gam, sigma2) for each validation method
[min_random, idx_random] = min(perf_random(:));
[min_kfold, idx_kfold] = min(perf_kfold(:));
[min_leaveoneout, idx_leaveoneout] = min(perf_leaveoneout(:));

[best_i_random, best_j_random] = ind2sub(size(perf_random), idx_random);
[best_i_kfold, best_j_kfold] = ind2sub(size(perf_kfold), idx_kfold);
[best_i_leaveoneout, best_j_leaveoneout] = ind2sub(size(perf_leaveoneout), idx_leaveoneout);

fprintf('\nBest gamma/sigma² (Random Split) = %.4f / %.4f\n', gam_values(best_i_random), sigma2_values(best_j_random));
fprintf('Best gamma/sigma² (10-Fold CV) = %.4f / %.4f\n', gam_values(best_i_kfold), sigma2_values(best_j_kfold));
fprintf('Best gamma/sigma² (Leave-One-Out) = %.4f / %.4f\n', gam_values(best_i_leaveoneout), sigma2_values(best_j_leaveoneout));

method_names = {'Random Split', '10-Fold CV', 'Leave-One-Out'};
best_params = [
    gam_values(best_i_random), sigma2_values(best_j_random);
    gam_values(best_i_kfold), sigma2_values(best_j_kfold);
    gam_values(best_i_leaveoneout), sigma2_values(best_j_leaveoneout);
];

% Evaluate best models on the test set
for k = 1:3
    gam_best = best_params(k, 1);
    sigma2_best = best_params(k, 2);

    % Train LS-SVM on the full training data with the selected parameters
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam_best, sigma2_best, 'RBF_kernel'});

    % Predict on the test set
    [Yht, Zt] = simlssvm({Xtrain, Ytrain, type, gam_best, sigma2_best, 'RBF_kernel'}, {alpha, b}, Xtest);

    % Compute misclassification rate
    err = sum(Yht ~= Ytest);
    err_rate = err / length(Ytest) * 100;

    % Display result
    fprintf('Test set performance (%s): #misclassified = %d, error rate = %.2f%%\n', ...
        method_names{k}, err, err_rate);
end


for k = 1:3
    gam_best = best_params(k, 1);
    sigma2_best = best_params(k, 2);

    % Train the LS-SVM with best hyperparameters
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam_best, sigma2_best, 'RBF_kernel'});

    % Plot decision boundary
    figure;
    plotlssvm({Xtrain, Ytrain, type, gam_best, sigma2_best, 'RBF_kernel', 'preprocess'}, {alpha, b});
    title(['LS-SVM Decision Boundary - ', method_names{k}, ...
           ' (\gamma = ', num2str(gam_best), ', \sigma^2 = ', num2str(sigma2_best), ')']);
end

%%

% === SETTINGS ===
type = 'c';                      % Classification
kernel_type = 'RBF_kernel';     % RBF kernel
gam_range = logspace(-3, 3, 50);  
sig2_range = logspace(-3, 3, 50);  

% === TUNING WITH SIMPLEX ===
disp('Tuning with SIMPLEX...');
tic;
[gam_simplex, sig2_simplex, cost_simplex] = tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type}, ...
    'simplex', 'crossvalidatelssvm', {10, 'misclass'});
time_simplex = toc;
fprintf('Simplex tuning results:\n');
fprintf('gamma = %.4f, sigma² = %.4f, cost = %.4f, time = %.2fs\n\n', ...
    gam_simplex, sig2_simplex, cost_simplex, time_simplex);

% === TUNING WITH GRIDSEARCH ===
disp('Tuning with GRIDSEARCH (fine grid)...');
global GRID_GAMMA GRID_SIGMA2;
GRID_GAMMA = gam_range;
GRID_SIGMA2 = sig2_range;
tic;
[gam_grid, sig2_grid, cost_grid] = tunelssvm({Xtrain, Ytrain, type, [], [], kernel_type}, ...
    'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
time_grid = toc;
fprintf('Gridsearch tuning results:\n');
fprintf('gamma = %.4f, sigma² = %.4f, cost = %.4f, time = %.2fs\n\n', ...
    gam_grid, sig2_grid, cost_grid, time_grid);

% === COMPARE RESULTS ===
disp('--- Summary ---');
disp(['Simplex: gamma = ', num2str(gam_simplex), ', sigma² = ', num2str(sig2_simplex), ...
      ', cost = ', num2str(cost_simplex), ', time = ', num2str(time_simplex), ' seconds']);
disp(['Gridsearch: gamma = ', num2str(gam_grid), ', sigma² = ', num2str(sig2_grid), ...
      ', cost = ', num2str(cost_grid), ', time = ', num2str(time_grid), ' seconds']);

selected_gamma = gam_grid;
selected_sig2 = sig2_grid;


%%

% Step 1: Train the classifier on the training set
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'c', selected_gamma, selected_sig2, 'RBF_kernel'});

% Step 2: Classify the test set
[Ytest_pred, Ylatent] = simlssvm({Xtrain, Ytrain, 'c', selected_gamma, selected_sig2, 'RBF_kernel'}, {alpha, b}, Xtest);

% Step 3: Generate and plot the ROC curve
roc(Ylatent, Ytest);
title('ROC Curve on Test Set using Tuned LS-SVM (RBF Kernel)');

%%

% Use bayesian modoutclass function to visualize probabilities
bay_modoutClass({Xtrain, Ytrain, 'c', selected_gamma, selected_sig2}, 'figure');

% Add colorbar to the figure to interpret probabilities
colorbar;
title('Bayesian Posterior Probability Estimates');

%%

% Define ranges for gamma and sigma^2
gam_values = [10, 1000];  
sigma2_values = [0.1, 10];  

% Loop through combinations and visualize
figure_idx = 1;
for i = 1:length(gam_values)
    for j = 1:length(sigma2_values)
        gam = gam_values(i);
        sig2 = sigma2_values(j);

        figure(figure_idx);
        bay_modoutClass({Xtrain, Ytrain, 'c', gam, sig2}, 'figure');
        colorbar;
        title(sprintf('Bayesian Posterior (\\gamma = %.2g, \\sigma^2 = %.2g)', gam, sig2));
        
        figure_idx = figure_idx + 1;
    end
end


