data = load('shuttle.dat','-ascii'); function_type = 'c';

data = data(1:1000,:);

X = data(:,1:end-1);
Y = data(:,end);

% binarize the labels
Y(Y == 1) = 1;
Y(Y ~= 1) = -1;

testX = [];
testY = [];

%%

% Basic statistics
num_samples = size(X, 1);
num_features = size(X, 2);
unique_classes = unique(Y);
num_classes = numel(unique_classes);

fprintf('Number of datapoints: %d\n', num_samples);
fprintf('Number of attributes: %d\n', num_features);
fprintf('Number of classes: %d\n', num_classes);
fprintf('Class distribution:\n');
tabulate(Y)

% Class balance visualization
figure;
histogram(Y, 'BinMethod', 'integers');
title('Class Distribution');
xlabel('Class Label');
ylabel('Frequency');
grid on;

%%
% Feature distributions
figure;
for i = 1:min(9, num_features)
    subplot(3, 3, i);
    histogram(X(:, i));
    title(sprintf('Feature %d', i));
    xlabel('Value');
    ylabel('Count');
end
sgtitle('Feature Distributions');

% PCA for visualization

% Standardize features
X_std = zscore(X);

% Perform PCA
[coeff, score] = pca(X_std);

% PCA plot (after standardization)
figure;
gscatter(score(:,1), score(:,2), Y);
title('PCA Projection of Shuttle Dataset (Standardized)');
xlabel('PC1'); ylabel('PC2');
legend('show');
grid on;
xlim([-3 8]);
ylim([-5 5]);
xticks(-3:1:8);
yticks(-5:1:5);

%%

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

