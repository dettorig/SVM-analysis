%% Data Preparation: Split Train/Val
N = length(Z);
train_frac = 0.75;

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

%% Grid Search Based on Recursive Validation Performance

gam_values  = [0.1, 1, 10, 100, 1000];
sig2_values = [0.01, 1, 10, 100, 1000];
order_values = 1:5:51;

cv_bestparams.order = [];
cv_bestparams.gam = [];
cv_bestparams.sig2 = [];
cv_bestparams.cv_mse = [];

best_error = inf;
nb_val = 100;

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

%% Recursive Prediction Using Best Parameters from Grid Search

order = cv_bestparams.order;
gam = cv_bestparams.gam;
sig2 = cv_bestparams.sig2;

Ztrainval = [Ztrain; Zval];
Xtrain = windowize(Ztrainval, 1:(order+1));
Ytrain = Xtrain(:, end);
Xtrain = Xtrain(:, 1:order);

task_best = {Xtrain, Ytrain, 'function estimation', gam, sig2, 'RBF_kernel'};
[alpha_best, b_best] = trainlssvm(task_best);

nb = 200;
x_input = Ztrainval(end - order + 1:end);
Ytest_pred = zeros(nb, 1);

for t = 1:nb
    y_next = simlssvm(task_best, {alpha_best, b_best}, x_input(:)');
    Ytest_pred(t) = y_next;
    x_input = [x_input(2:end); y_next];
end

test_mse = mean((Ztest(1:nb) - Ytest_pred).^2);

fprintf('\nBest parameters found:\n');
fprintf('Order = %d\n', cv_bestparams.order);
fprintf('Gamma = %.4f\n', cv_bestparams.gam);
fprintf('Sigma^2 = %.4f\n', cv_bestparams.sig2);
fprintf('Recursive Val MSE = %.4f\n', cv_bestparams.cv_mse);
fprintf('Test MSE = %.4f\n', test_mse);

%% Plotting Final Prediction

figure;
plot(Ztest(1:nb), 'k', 'LineWidth', 1.5);
hold on;
plot(Ytest_pred, 'r', 'LineWidth', 1.5);
legend('Test Data', 'Predictions');
title('LS-SVM Recursive Prediction with Validation-Based Grid Search');
xlabel('Time Index');
ylabel('Value');
grid on;
hold off;
