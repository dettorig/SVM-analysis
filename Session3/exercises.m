clc;
clear;

%% 1. Generate the structured dataset
nb = 400;
sig = 0.3;
nb = nb / 2;

for t = 1:nb
    yin(t,:)  = [2*sin(t/nb*pi),   2*cos(0.61*t/nb*pi), (t/nb*sig)];
    yang(t,:) = [-2*sin(t/nb*pi),  0.45 - 2*cos(0.61*t/nb*pi), (t/nb*sig)];
    samplesyin(t,:)  = [yin(t,1) + yin(t,3)*randn,  yin(t,2) + yin(t,3)*randn];
    samplesyang(t,:) = [yang(t,1) + yang(t,3)*randn, yang(t,2) + yang(t,3)*randn];
end

X = [samplesyin; samplesyang];
N = size(X, 1);

%% 2. Define parameter grids
sig2_values = [0.01, 1, 10, 100, 1000];
max_nc   = 20;

%% 3. Reconstruction error plots vs number of components

approx_type = 'eigs';  % Lanczos approximation
figure;

for i = 1:length(sig2_values)
    sig2 = sig2_values(i);
    recon_errors = zeros(max_nc, 1);

    for nc = 1:max_nc
        try
            X_rec = denoise_kpca(X, 'RBF_kernel', sig2, [], approx_type, nc);
            recon_errors(nc) = mean(sum((X - X_rec).^2, 2));  % MSE
        catch ME
            recon_errors(nc) = NaN;
            fprintf('[ERROR] σ²=%.2f, nc=%d: %s\n', sig2, nc, ME.message);
        end
    end

    subplot(2, ceil(length(sig2_values)/2), i);
    plot(1:max_nc, recon_errors, 'r-o', 'LineWidth', 1.5);
    xlabel('Number of Components');
    ylabel('Reconstruction Error (MSE)');
    title(sprintf('\\sigma^2 = %.2f', sig2));
    ylim([0 max(recon_errors(~isnan(recon_errors))) * 1.1]);
    grid on;
end

sgtitle('Reconstruction Error vs Number of Components for Different \sigma^2');


%% 5. Final model with best parameters

best_sig2 = 1;
best_nc = 4;
xd = denoise_kpca(X, 'RBF_kernel', best_sig2, [], 'eigs', best_nc);

figure;
plot(samplesyin(:,1), samplesyin(:,2), 'bo'); hold on;
plot(samplesyang(:,1), samplesyang(:,2), 'go');
plot(xd(:,1), xd(:,2), 'r+');
title(sprintf('Kernel PCA Denoising\n(σ² = %.2f, nc = %d)', best_sig2, best_nc));
xlabel('X_1'); ylabel('X_2');
legend('Yin', 'Yang', 'Denoised');
grid on;

%% Kernel Spectral Clustering 

% clear;

% load two3drings; % load the toy example
[N, d] = size(X);

perm = randperm(N); % shuffle the data
X = X(perm, :);

sig2_values = [0.001, 0.005, 0.01, 0.02, 0.1, 0.2, 1];

for s = 1:length(sig2_values)
    sig2 = sig2_values(s);
    
    fprintf('\n--- sigma^2 = %.4f ---\n', sig2);

    % Compute the RBF kernel (affinity) matrix
    K = kernel_matrix(X, 'RBF_kernel', sig2);

    % Compute the degree matrix
    D = diag(sum(K));

    % Compute the top 3 eigenvectors of the random walk matrix
    [U, lambda] = eigs(inv(D) * K, 3);

    % Use the 2nd eigenvector for binary clustering
    clust = sign(U(:, 2));

    % Sort data using the clustering result
    [~, order] = sort(clust, 'descend');
    Xsorted = X(order, :);
    Ksorted = kernel_matrix(Xsorted, 'RBF_kernel', sig2);

    % Compute the projection using the 2nd and 3rd eigenvectors
    proj = K * U(:, 2:3);

    % Plotting
    figure('Name', sprintf('Spectral Clustering, sigma^2 = %.4f', sig2));

    subplot(2, 2, 1);
    scatter3(X(:, 1), X(:, 2), X(:, 3), 15);
    title('Original 3D Data');

    subplot(2, 2, 2);
    scatter3(X(:, 1), X(:, 2), X(:, 3), 30, clust);
    title('Clustering Result');

    subplot(2, 2, 3);
    imshow(K, []);
    title('Original Kernel Matrix');

    subplot(2, 2, 4);
    imshow(Ksorted, []);
    title('Sorted Kernel Matrix');
    
    figure('Name', sprintf('Projection, sigma^2 = %.4f', sig2));
    scatter(proj(:,1), proj(:,2), 15, clust);
    title('Projection onto Spectral Subspace');

    pause; % Wait for user before continuing to next sigma^2
end

%%
