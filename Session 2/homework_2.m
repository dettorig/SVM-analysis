%% Recursive Prediction with Fixed Order and Parameters

order = 10;
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

gam = 10;
sig2 = 10;
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

% Recursive prediction for nb future points
nb = 200;
x_input = Z(end - order + 1:end);  % last known input window
prediction = zeros(nb, 1);

for t = 1:nb
    y_next = simlssvm({X, Y, 'f', gam, sig2}, {alpha, b}, x_input(:)');
    prediction(t) = y_next;
    x_input = [x_input(2:end); y_next];  % shift window
end

% Compute MSE on test set
test_mse = mean((Ztest(1:nb) - prediction).^2);

% Display result
fprintf('Test MSE over %d predicted steps: %.4f\n', nb, test_mse);

figure;
hold on;
plot(Ztest(1:nb), 'k', 'LineWidth', 1.5);
plot(prediction, 'r', 'LineWidth', 1.5);
legend('Actual', 'Prediction');
title('LS-SVM Recursive Forecast - Order 10');
xlabel('Time Step');
ylabel('Value');
hold off;

%% Grid Search Cross-Validation

gam_values  = [0.1, 1, 10, 100, 1000];
sig2_values = [0.01, 1, 10, 100, 1000];
order_values = 1:5:51;

cv_bestparams.order = [];
cv_bestparams.gam = [];
cv_bestparams.sig2 = [];
cv_bestparams.cv_mse = [];

best_error = inf;
K = 10;

for ord_idx = 1:length(order_values)
    order = order_values(ord_idx);

    X_full = windowize(Z, 1:(order+1));
    Y_full = X_full(:, end);
    X_full = X_full(:, 1:order);

    cv = cvpartition(length(Y_full), 'KFold', K);

    for i_gam = 1:length(gam_values)
        gam = gam_values(i_gam);

        for j_sig2 = 1:length(sig2_values)
            sig2 = sig2_values(j_sig2);

            mse_folds = zeros(K, 1);

            for fold = 1:K
                train_idx = training(cv, fold);
                test_idx = test(cv, fold);

                Xtrain = X_full(train_idx, :);
                Ytrain = Y_full(train_idx);
                Xcv = X_full(test_idx, :);
                Ycv = Y_full(test_idx);

                task = {Xtrain, Ytrain, 'function estimation', gam, sig2, 'RBF_kernel'};
                [alpha, b] = trainlssvm(task);

                Ycv_pred = simlssvm(task, {alpha, b}, Xcv);
                mse_folds(fold) = mean((Ycv - Ycv_pred).^2);
            end

            cv_error = mean(mse_folds);

            if cv_error < best_error
                best_error = cv_error;
                cv_bestparams.order = order;
                cv_bestparams.gam = gam;
                cv_bestparams.sig2 = sig2;
                cv_bestparams.cv_mse = cv_error;
            end

            fprintf('Order=%d, Gamma=%.4e, Sigma^2=%.4e, CV MSE=%.4f\n', ...
                order, gam, sig2, cv_error);
        end
    end
end

%% Recursive Prediction Using Best Parameters from Grid Search

order = cv_bestparams.order;
gam = cv_bestparams.gam;
sig2 = cv_bestparams.sig2;

Xtrain = windowize(Z, 1:(order+1));
Ytrain = Xtrain(:, end);
Xtrain = Xtrain(:, 1:order);

task_best = {Xtrain, Ytrain, 'function estimation', gam, sig2, 'RBF_kernel'};
[alpha_best, b_best] = trainlssvm(task_best);

% Recursive forecasting
nb = 200;
x_input = Z(end - order + 1:end);  % last training values
Ytest_pred = zeros(nb, 1);

for t = 1:nb
    y_next = simlssvm(task_best, {alpha_best, b_best}, x_input(:)');
    Ytest_pred(t) = y_next;
    x_input = [x_input(2:end); y_next];
end

% Compute test MSE (compare to actual future values)
test_mse = mean((Ztest(1:nb) - Ytest_pred).^2);

% Display results
fprintf('\nBest parameters found:\n');
fprintf('Order = %d\n', cv_bestparams.order);
fprintf('Gamma = %.4f\n', cv_bestparams.gam);
fprintf('Sigma^2 = %.4f\n', cv_bestparams.sig2);
fprintf('Cross-validated MSE = %.4f\n', cv_bestparams.cv_mse);
fprintf('Test MSE = %.4f\n', test_mse);

%% Plotting Final Prediction

figure;
plot(Ztest(1:nb), 'k', 'LineWidth', 1.5);
hold on;
plot(Ytest_pred, 'r', 'LineWidth', 1.5);
legend('Test Data', 'Predictions');
title('LS-SVM Recursive Prediction with Grid Search Parameters');
xlabel('Time Index');
ylabel('Value');
grid on;
hold off;
