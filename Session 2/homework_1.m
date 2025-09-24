%% Data Preparation: Split into Train/Val
N = length(Z);
train_frac = 0.70;
Ntrain = floor(N * train_frac);

Ztrain = Z(1:Ntrain);
Zval = Z(Ntrain+1:end);

%% Recursive Validation Function
function val_mse = recursive_val_error(Ztrain, Zval, order, gam, sig2, nb)
    Xtrain = windowize(Ztrain, 1:(order + 1));
    Ytrain = Xtrain(:, end);
    Xtrain = Xtrain(:, 1:order);

    task = {Xtrain, Ytrain, 'function estimation', gam, sig2, 'RBF_kernel'};
    [alpha, b] = trainlssvm(task);

    x_input = Ztrain(end - order + 1:end);
    prediction = zeros(nb, 1);
    for t = 1:nb
        y_next = simlssvm(task, {alpha, b}, x_input(:)');
        prediction(t) = y_next;
        x_input = [x_input(2:end); y_next];
    end

    val_mse = mean((Zval(1:nb) - prediction).^2);
end

%% Recursive Validation-Based Grid Search

gam_values  = [0.1, 1, 10, 100, 1000];
sig2_values = [0.01, 1, 10, 100, 1000];
order_values = 1:15;

cv_bestparams.order = [];
cv_bestparams.gam = [];
cv_bestparams.sig2 = [];
cv_bestparams.cv_mse = [];

best_error = inf;
nb_val = 45;  % number of validation steps used in recursive forecast

for ord_idx = 1:length(order_values)
    order = order_values(ord_idx);

    for i_gam = 1:length(gam_values)
        gam = gam_values(i_gam);

        for j_sig2 = 1:length(sig2_values)
            sig2 = sig2_values(j_sig2);

            val_mse = recursive_val_error(Ztrain, Zval, order, gam, sig2, nb_val);

            if val_mse < best_error
                best_error = val_mse;
                cv_bestparams.order = order;
                cv_bestparams.gam = gam;
                cv_bestparams.sig2 = sig2;
                cv_bestparams.cv_mse = val_mse;
            end

            fprintf('Order=%d, Gamma=%.4e, Sigma^2=%.4e, Recursive Val MSE=%.4f\n', ...
                    order, gam, sig2, val_mse);
        end
    end
end

% Display best parameters
fprintf('\nBest parameters found:\n');
fprintf('Order = %d\n', cv_bestparams.order);
fprintf('Gamma = %.4f\n', cv_bestparams.gam);
fprintf('Sigma^2 = %.4f\n', cv_bestparams.sig2);
fprintf('Recursive Val MSE = %.4f\n', cv_bestparams.cv_mse);

% Save only the best parameters
save('best_cv_params.mat', 'cv_bestparams');


%% Bayesian Optimization with Order as Hyperparameter

orders = 1:15;  % Candidate orders
best_crit = inf;

gam_init = 10;
sig2_init = 10;

bay_bestparams = struct();
bay_bestparams.order = [];
bay_bestparams.gam = [];
bay_bestparams.sig2 = [];
bay_bestparams.alpha = [];
bay_bestparams.b = [];
bay_bestparams.criterion = [];

for order = orders
    X = windowize(Z, 1:(order+1));
    Y = X(:, end);
    X = X(:, 1:order);

    crit_order = bay_lssvm({X, Y, 'f', gam_init, sig2_init}, 1);

    fprintf('Order %d: Bayesian criterion = %.4f\n', order, crit_order);

    if crit_order < best_crit
        best_crit = crit_order;
        bay_bestparams.order = order;
    end
end

order = bay_bestparams.order;
X = windowize(Z, 1:(order+1));
Y = X(:, end);
X = X(:, 1:order);

[~, alpha, b] = bay_optimize({X, Y, 'f', gam_init, sig2_init}, 1);
bay_bestparams.alpha = alpha;
bay_bestparams.b = b;

[~, gam_opt] = bay_optimize({X, Y, 'f', gam_init, sig2_init}, 2);
bay_bestparams.gam = gam_opt;

[~, sig2_opt] = bay_optimize({X, Y, 'f', gam_opt, sig2_init}, 3);
bay_bestparams.sig2 = sig2_opt;

final_crit = bay_lssvm({X, Y, 'f', gam_opt, sig2_opt}, 1);
bay_bestparams.criterion = final_crit;

fprintf('\n================= Best Model Found =================\n');
fprintf('Order = %d\n', bay_bestparams.order);
fprintf('Gamma = %.4f\n', bay_bestparams.gam);
fprintf('Sigma^2 = %.4f\n', bay_bestparams.sig2);
fprintf('Bayesian criterion = %.4f\n', final_crit);
fprintf('=====================================================\n');

save('bay_bestparams.mat', 'bay_bestparams');

%% Load Best Parameters
load('best_cv_params.mat');
load('bay_bestparams.mat');
%%
nb = 50;

%%
gam_cv = cv_bestparams.gam(end);
sig2_cv = cv_bestparams.sig2(end);
order_cv = cv_bestparams.order(end);

Xtrain_cv = windowize(Z, 1:(order_cv+1));
Ytrain_cv = Xtrain_cv(:, end);
Xtrain_cv = Xtrain_cv(:, 1:order_cv);

%%

gam_bay = bay_bestparams.gam(end);
sig2_bay = bay_bestparams.sig2(end);
order_bay = bay_bestparams.order(end);

Xtrain_bay = windowize(Z, 1:(order_bay+1));
Ytrain_bay = Xtrain_bay(:, end);
Xtrain_bay = Xtrain_bay(:, 1:order_bay);

%% Inference with Grid Search Parameters - Recursive

task_cv = {Xtrain_cv, Ytrain_cv, 'function estimation', gam_cv, sig2_cv, 'RBF_kernel'};
[alpha_cv, b_cv] = trainlssvm(task_cv);

x_input_cv = Z(end-order_cv+1:end);
Ytest_pred_cv = zeros(nb, 1);

for t = 1:nb
    y_next = simlssvm(task_cv, {alpha_cv, b_cv}, x_input_cv(:)');
    Ytest_pred_cv(t) = y_next;
    x_input_cv = [x_input_cv(2:end); y_next];
end

%% Inference with Bayesian Parameters - Recursive

task_bay = {Xtrain_bay, Ytrain_bay, 'function estimation', gam_bay, sig2_bay, 'RBF_kernel'};
[alpha_bay, b_bay] = trainlssvm(task_bay);

x_input_bay = Z(end-order_bay+1:end);
Ytest_pred_bay = zeros(nb, 1);

for t = 1:nb
    y_next = simlssvm(task_bay, {alpha_bay, b_bay}, x_input_bay(:)');
    Ytest_pred_bay(t) = y_next;
    x_input_bay = [x_input_bay(2:end); y_next];
end


%% Plotting - Grid Search Estimator
figure;
hold on;
plot(Ztest(1:nb), 'k', 'LineWidth', 1.5);
plot(Ytest_pred_cv, 'b', 'LineWidth', 1.5);
legend('Actual', 'Grid Search Prediction');
title('LS-SVM - Grid Search Estimator (Recursive Forecasting)');
xlabel('Time Step');
ylabel('Value');
grid on;
hold off;

%% Plotting - Bayesian Estimator
figure;
hold on;
plot(Ztest(1:nb), 'k', 'LineWidth', 1.5);
plot(Ytest_pred_bay, 'r', 'LineWidth', 1.5);
legend('Actual', 'Bayesian Prediction');
title('LS-SVM - Bayesian Estimator (Recursive Forecasting)');
xlabel('Time Step');
ylabel('Value');
grid on;
hold off;

