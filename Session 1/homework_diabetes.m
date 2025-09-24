% total is N x 8, labels_total is N x 1

% === Standardize the data ===
total_std = zscore(total);  % standardizes each column (feature) to mean 0, std 1

% === Perform PCA on standardized data ===
[coeff, score, latent] = pca(total_std);

% === Compute variance explained by each component ===
explained_variance = latent / sum(latent);
var_pc1 = explained_variance(1);
var_pc2 = explained_variance(2);
total_var_12 = var_pc1 + var_pc2;

% === Display variance explained ===
fprintf('Variance explained by PC1: %.2f%%\n', var_pc1 * 100);
fprintf('Variance explained by PC2: %.2f%%\n', var_pc2 * 100);
fprintf('Total variance explained by first two PCs: %.2f%%\n', total_var_12 * 100);

% === Plot PCA projection ===
figure;
gscatter(score(:,1), score(:,2), labels_total, 'rb', 'ox');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('2D PCA Projection of Standardized Diabetes Dataset');
legend('Class -1', 'Class +1');
grid on;


%%

sum(labels_train == +1) 
sum(labels_train == -1)

%%

% Assuming 'trainset' is N x D (e.g., 8D feature matrix)
corr_matrix = corr(trainset);

% Display the correlation matrix
disp('Correlation matrix among features:');
disp(corr_matrix);
corr_matrix = corr(trainset);

figure;
imagesc(corr_matrix);
colorbar;
caxis([-1, 1]);              % explicitly set color range to include negative values
colormap(jet);               % or 'coolwarm' if you have it
title('Feature Correlation Matrix');
xlabel('Feature index');
ylabel('Feature index');

%%

feature_vars = var(trainset);
low_variance_idx = find(feature_vars < 1e-5);
fprintf('Low-variance feature indices: %s\n', mat2str(low_variance_idx));

%%

for i = 1:size(trainset,2)
    figure;
    boxplot(trainset(:,i), labels_train);
    title(sprintf('Boxplot of Feature %d by Class', i));
    xlabel('Class'); ylabel('Feature Value');
end

%%
alpha_bonf = 0.05 / size(trainset, 2);  % Bonferroni-corrected threshold

for i = 1:size(trainset,2)
    p = anova1(trainset(:,i), labels_train, 'off');
    if p < alpha_bonf
        significance = 'significant';
    else
        significance = 'not significant';
    end
    fprintf('Feature %d: ANOVA p = %.4f (%s)\n', i, p, significance);
end




%%

% === Load and prepare data ===
labels_train = labels_train(:); 
labels_test = labels_test(:);

type = 'c';
cv_args = {10, 'misclass'};
errors = zeros(1,3);

% === Define shared tuning grid ===
global GRID_GAMMA GRID_SIGMA2;
GRID_GAMMA = logspace(-3, 3, 30);     % used in all kernels
GRID_SIGMA2 = logspace(-1, 1, 30);    % used only in RBF

% === 1. LINEAR KERNEL ===
disp('--- LINEAR LS-SVM ---');
model_init = initlssvm(trainset, labels_train, type, [], [], 'lin_kernel');
model_lin = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
[Y_pred_lin, Y_latent_lin] = simlssvm(model_lin, testset);
errors(1) = mean(Y_pred_lin ~= labels_test);

figure;
roc(Y_latent_lin, labels_test);
title(sprintf('Linear Kernel - ROC (Err: %.2f%%)', errors(1)*100));

% === 2. POLYNOMIAL KERNEL with tuned degree ===
disp('--- POLYNOMIAL LS-SVM ---');
degrees = [2, 3, 4];
best_poly_err = inf;

for d = degrees
    fprintf('Trying degree %d...\n', d);
    % Set kernel parameters [degree; offset] = [d; 1]
    model_init = initlssvm(trainset, labels_train, type, [], [d; 1], 'poly_kernel');
    model = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
    [Y_pred, Y_latent] = simlssvm(model, testset);
    err = mean(Y_pred ~= labels_test);
    
    if err < best_poly_err
        best_poly_err = err;
        best_model_poly = model;
        best_latent_poly = Y_latent;
        best_d = d;
    end
end

errors(2) = best_poly_err;

figure;
roc(best_latent_poly, labels_test);
title(sprintf('Polynomial Kernel (d=%d) - ROC (Err: %.2f%%)', best_d, errors(2)*100));

% === 3. RBF KERNEL ===
disp('--- RBF LS-SVM ---');
model_init = initlssvm(trainset, labels_train, type, [], [], 'RBF_kernel');
model_rbf = tunelssvm(model_init, 'gridsearch', 'crossvalidatelssvm', cv_args);
[Y_pred_rbf, Y_latent_rbf] = simlssvm(model_rbf, testset);
errors(3) = mean(Y_pred_rbf ~= labels_test);

figure;
roc(Y_latent_rbf, labels_test);
title(sprintf('RBF Kernel - ROC (Err: %.2f%%)', errors(3)*100));

% === Summary ===
fprintf('\n--- Classification Error Rates ---\n');
fprintf('Linear     : %.2f%%\n', errors(1)*100);
fprintf('Polynomial : %.2f%% (degree = %d)\n', errors(2)*100, best_d);
fprintf('RBF        : %.2f%%\n', errors(3)*100);

%%

function auc = compute_manual_auc(scores, labels)
    % Ensure column vectors
    scores = scores(:);
    labels = labels(:);

    % Sort by descending score
    [scores_sorted, sort_idx] = sort(scores, 'descend');
    labels_sorted = labels(sort_idx);

    % Binary classification assumption: labels in {+1, -1}
    P = sum(labels_sorted == 1);
    N = sum(labels_sorted == -1);

    if P == 0 || N == 0
        auc = NaN;
        warning('AUC undefined: dataset has only one class.');
        return;
    end

    % Compute TPR and FPR
    TPR = cumsum(labels_sorted == 1) / P;
    FPR = cumsum(labels_sorted == -1) / N;

    % Compute AUC using trapezoidal rule
    auc = trapz(FPR, TPR);
end

% === Manual AUC computation ===
auc_lin  = compute_manual_auc(Y_latent_lin, labels_test);
auc_poly = compute_manual_auc(best_latent_poly, labels_test);
auc_rbf  = compute_manual_auc(Y_latent_rbf, labels_test);

fprintf('\n--- Area Under ROC Curve (AUC) ---\n');
fprintf('Linear     : %.4f\n', auc_lin);
fprintf('Polynomial : %.4f\n', auc_poly);
fprintf('RBF        : %.4f\n', auc_rbf);

