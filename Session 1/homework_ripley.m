X_all = [Xtrain; Xtest];
Y_all = [Ytrain; Ytest];

figure;
gscatter(X_all(:,1), X_all(:,2), Y_all, 'rb', 'xo');
xlabel('X_1'); ylabel('X_2');
title('Ripley Dataset');
legend('Class -1', 'Class +1');
axis equal;
grid on;


%%


% === Load and prepare Ripley data ===
Ytrain = Ytrain(:); 
Ytest = Ytest(:);

type = 'c';
cv_args = {10, 'misclass'};
errors = zeros(1,3);

% === Define shared tuning grid ===
global GRID_GAMMA GRID_SIGMA2;
GRID_GAMMA = logspace(-3, 3, 30);     % used in all kernels
GRID_SIGMA2 = logspace(-3, 3, 30);    % used only in RBF

% === 1. LINEAR KERNEL ===
disp('--- LINEAR LS-SVM ---');
model_init = initlssvm(Xtrain, Ytrain, type, [], [], 'lin_kernel');
model_lin = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
[Y_pred_lin, Y_latent_lin] = simlssvm(model_lin, Xtest);
errors(1) = mean(Y_pred_lin ~= Ytest);

figure;
roc(Y_latent_lin, Ytest);
title(sprintf('Linear Kernel - ROC (Err: %.2f%%)', errors(1)*100));

% === 2. POLYNOMIAL KERNEL with tuned degree ===
disp('--- POLYNOMIAL LS-SVM ---');
degrees = [2, 3, 4];
best_poly_err = inf;

for d = degrees
    fprintf('Trying degree %d...\n', d);
    model_init = initlssvm(Xtrain, Ytrain, type, [], [d; 1], 'poly_kernel');
    model = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
    [Y_pred, Y_latent] = simlssvm(model, Xtest);
    err = mean(Y_pred ~= Ytest);
    
    if err < best_poly_err
        best_poly_err = err;
        best_model_poly = model;
        best_latent_poly = Y_latent;
        best_d = d;
    end
end

errors(2) = best_poly_err;

figure;
roc(best_latent_poly, Ytest);
title(sprintf('Polynomial Kernel (d=%d) - ROC (Err: %.2f%%)', best_d, errors(2)*100));

% === 3. RBF KERNEL ===
disp('--- RBF LS-SVM ---');
model_init = initlssvm(Xtrain, Ytrain, type, [], [], 'RBF_kernel');
model_rbf = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
[Y_pred_rbf, Y_latent_rbf] = simlssvm(model_rbf, Xtest);
errors(3) = mean(Y_pred_rbf ~= Ytest);

figure;
roc(Y_latent_rbf, Ytest);
title(sprintf('RBF Kernel - ROC (Err: %.2f%%)', errors(3)*100));

% === Summary ===
fprintf('\n--- Classification Error Rates (Ripley Dataset) ---\n');
fprintf('Linear     : %.2f%%\n', errors(1)*100);
fprintf('Polynomial : %.2f%% (degree = %d)\n', errors(2)*100, best_d);
fprintf('RBF        : %.2f%%\n', errors(3)*100);

%%

% === Helper function to compute ROC and AUC manually ===
function auc = compute_manual_auc(scores, labels)
    [scores_sorted, sort_idx] = sort(scores, 'descend');
    labels_sorted = labels(sort_idx);

    P = sum(labels_sorted == 1);
    N = sum(labels_sorted == -1);

    TPR = cumsum(labels_sorted == 1) / P;  % True Positive Rate
    FPR = cumsum(labels_sorted == -1) / N; % False Positive Rate

    auc = trapz(FPR, TPR);
end

% === Compute AUCs for each kernel
auc_lin  = compute_manual_auc(Y_latent_lin, Ytest);
auc_poly = compute_manual_auc(best_latent_poly, Ytest);
auc_rbf  = compute_manual_auc(Y_latent_rbf, Ytest);

% === Print results
fprintf('\n--- Area Under ROC Curve (AUC) ---\n');
fprintf('Linear     : %.4f\n', auc_lin);
fprintf('Polynomial : %.4f\n', auc_poly);
fprintf('RBF        : %.4f\n', auc_rbf);

