% Load and prepare California Housing dataset
data = load('california.dat','-ascii');
function_type = 'f';
data = data(1:1000,:)

% Split into predictors and target
X = data(:, 1:end-1);
Y = data(:, end);

testX = [];
testY = [];

%% Basic statistics
num_samples = size(X, 1);
num_features = size(X, 2);

fprintf('Number of datapoints: %d\n', num_samples);
fprintf('Number of attributes: %d\n', num_features);

%%
figure;
histogram(Y, 50);  % use 50 bins instead of 'integers'
title('Target Distribution (Median House Value)');
xlabel('Value');
ylabel('Frequency');
grid on;

%% Feature distributions
% Feature distributions (consistent histogram style)
figure;
for i = 1:min(9, num_features)
    subplot(3, 3, i);
    histogram(X(:, i), 30, ...                % fixed number of bins
        'FaceColor', [0.2 0.6 0.8], ...       % consistent color (light blue)
        'EdgeColor', 'black');               % black edges
    title(sprintf('Feature %d', i));
    xlabel('Value');
    ylabel('Count');
    grid on;
end
sgtitle('Feature Distributions (California Housing)');


%%
% Standardize features
X_std = zscore(X);

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  
X_std = zscore(X);
k = 5;
%function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'WINDOW'};
window = [15,20,25];

[process_matrix_err, process_matrix_sv, process_matrix_time] = fslssvm(X_std,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

%%

% Remove capped median house values (known issue in dataset)
cap_value = 500001;
cap_mask = Y < cap_value;

% Count and report number of removed samples
num_removed = sum(Y == cap_value);
fprintf('Removed %d capped samples (Y == %d)\n', num_removed, cap_value);

%%
X = X(cap_mask, :);
Y = Y(cap_mask);

% Standardize features
X_std = zscore(X);

% Parameter for input space selection
k = 5;
function_type = 'f';  % regression
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

% Process to be performed
user_process = {'WINDOW'};
window = [15,20,25];

% Optional test data (left empty)
testX = [];
testY = [];

% Run FS-LSSVM
[process_matrix_err, process_matrix_sv, process_matrix_time] = ...
    fslssvm(X_std, Y, k, function_type, kernel_type, global_opt, user_process, window, testX, testY);
