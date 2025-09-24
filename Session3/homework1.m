% Investigation of MSE over different sigmafactor values

rng('default');  % Restore modern RNG with 'twister' algorithm

% === Setup ===
[N, dim] = size(X);
Ntest1 = size(Xtest1, 1);
Ntest2 = size(Xtest2, 1);
maxx = max(X(:));
    
% === Add Gaussian noise ===
noisefactor = 0.3;
noise_std = noisefactor * maxx;

Xn = X;
for i = 1:N
    rng(i);
    Xn(i,:) = X(i,:) + noise_std * randn(1, dim);
end

Xnt1 = Xtest1;
for i = 1:Ntest1
    rng(N + i);
    Xnt1(i,:) = Xtest1(i,:) + noise_std * randn(1, dim);
end

Xnt2 = Xtest2;
for i = 1:Ntest2
    rng(N + Ntest1 + i);
    Xnt2(i,:) = Xtest2(i,:) + noise_std * randn(1, dim);
end

% === Fixed parameters ===
npcs_list = [2.^(0:7), 190];
sigmafactor_list = [0.8, 0.9, 1, 1.1]; 
base_sig2 = dim * mean(var(X));

mse_train = zeros(length(sigmafactor_list), length(npcs_list));
mse_val1  = zeros(size(mse_train));
mse_val2  = zeros(size(mse_train));


%%

for i = 1:length(sigmafactor_list)
    sf = sigmafactor_list(i);
    sig2 = base_sig2 * sf;
    fprintf('\n=== sigmafactor = %.4f (sig2 = %.4f) ===\n', sf, sig2);

    % Perform kernel PCA
    [lam, U] = kpca(X, 'RBF_kernel', sig2, [], 'eig', 240);
    [lam, idx] = sort(-lam); lam = -lam; U = U(:, idx);

    for j = 1:length(npcs_list)
        nb_pcs = npcs_list(j);
        Ud = U(:, 1:nb_pcs);

        % Reconstruct training set
        Xr_tr = zeros(size(X));
        for n = 1:N
            Xr_tr(n,:) = preimage_rbf(X, sig2, Ud, Xn(n,:), 'denoise');
        end

        % Reconstruct validation set 1
        Xr_v1 = zeros(size(Xtest1));
        for n = 1:Ntest1
            Xr_v1(n,:) = preimage_rbf(X, sig2, Ud, Xnt1(n,:), 'denoise');
        end

        % Reconstruct validation set 2
        Xr_v2 = zeros(size(Xtest2));
        for n = 1:Ntest2
            Xr_v2(n,:) = preimage_rbf(X, sig2, Ud, Xnt2(n,:), 'denoise');
        end

        % Compute and store MSEs
        mse_train(i, j) = mean((Xr_tr - X).^2, 'all');
        mse_val1(i, j)  = mean((Xr_v1 - Xtest1).^2, 'all');
        mse_val2(i, j)  = mean((Xr_v2 - Xtest2).^2, 'all');

        fprintf('npcs = %3d → MSE: train = %.4f, val1 = %.4f, val2 = %.4f\n', ...
                nb_pcs, mse_train(i,j), mse_val1(i,j), mse_val2(i,j));
    end
end

%%
% === Select best configuration (lowest validation MSE on Xtest1) ===
[min_val1, best_idx] = min(mse_val1(:));
[best_i, best_j] = ind2sub(size(mse_val1), best_idx);
best_sf = sigmafactor_list(best_i);
best_npcs = npcs_list(best_j);

fprintf('\n✅ Best validation MSE = %.4f at sigmafactor = %.4f, npcs = %d\n', ...
        min_val1, best_sf, best_npcs);

%%

% === Denoising visualization on validation set 1 (Xtest1) ===

ndig = 10;                     % Show digits 0–9
digs = 0:9;                    % Indexes for test digits
Xdt = zeros(ndig, dim);        % Denoised outputs
Xtr = X;                       % Use full training set

% Get best params from grid search
best_sig2 = base_sig2 * best_sf;
[lam, U] = kpca(X, 'RBF_kernel', best_sig2, [], 'eig', 240);
[lam, idx] = sort(-lam); lam = -lam; U = U(:, idx);

% Number of components to try
npcs_list = [2.^(0:7), 190];
lpcs = length(npcs_list);

figure;
colormap('gray');
sgtitle(sprintf('Denoising using kernel PCA — sigmafactor %.1f (Validation Set 1)', best_sf));

for k = 1:lpcs
    nb_pcs = npcs_list(k);
    fprintf('nb_pcs = %d\n', nb_pcs);
    Ud = U(:, 1:nb_pcs);

    for i = 1:ndig
        dig = digs(i);
        xt = Xnt1(i,:);  % Noisy validation digit
        if k == 1
            % Clean digit
            subplot(2 + lpcs, ndig, i);
            pcolor(1:15, 16:-1:1, reshape(Xtest1(i,:), 15, 16)'); shading interp;
            set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
            if i==1, ylabel('original'); end

            % Noisy digit
            subplot(2 + lpcs, ndig, i + ndig);
            pcolor(1:15, 16:-1:1, reshape(xt, 15, 16)'); shading interp;
            set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
            if i==1, ylabel('noisy'); end
        end

        % Denoised with current nb_pcs
        Xdt(i,:) = preimage_rbf(Xtr, best_sig2, Ud, xt, 'denoise');
        subplot(2 + lpcs, ndig, i + (2 + k - 1) * ndig);
        pcolor(1:15, 16:-1:1, reshape(Xdt(i,:), 15, 16)'); shading interp;
        set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
        if i==1, ylabel(['n=', num2str(nb_pcs)]); end
    end
end

%%
% === Select best configuration (lowest validation MSE on Xtest2) ===
[min_val2, best_idx2] = min(mse_val2(:));
[best_i2, best_j2] = ind2sub(size(mse_val2), best_idx2);
best_sf2 = sigmafactor_list(best_i2);
best_npcs2 = npcs_list(best_j2);

fprintf('\n✅ Best validation MSE on Xtest2 = %.4f at sigmafactor = %.4f, npcs = %d\n', ...
        min_val2, best_sf2, best_npcs2);

%%

% === Denoising visualization on validation set 2 (Xtest2) ===

ndig = 10;                     % Show digits 0–9
digs = 0:9;                    % Indexes for test digits
Xdt = zeros(ndig, dim);        % Denoised outputs
Xtr = X;                       % Use full training set

% Use best parameters from previous search
best_sig2 = base_sig2 * best_sf2;
[lam, U] = kpca(X, 'RBF_kernel', best_sig2, [], 'eig', 240);
[lam, idx] = sort(-lam); lam = -lam; U = U(:, idx);

% Number of components to try
npcs_list = [2.^(0:7), 190];
lpcs = length(npcs_list);

figure;
colormap('gray');
sgtitle(sprintf('Denoising using kernel PCA — sigmafactor %.1f (Validation Set 2)', best_sf2));  % ← fixed here

for k = 1:lpcs
    nb_pcs = npcs_list(k);
    fprintf('nb_pcs = %d\n', nb_pcs);
    Ud = U(:, 1:nb_pcs);

    for i = 1:ndig
        dig = digs(i);
        xt = Xnt2(i,:);  % Noisy digit from Xtest2
        if k == 1
            % Clean digit
            subplot(2 + lpcs, ndig, i);
            pcolor(1:15, 16:-1:1, reshape(Xtest2(i,:), 15, 16)'); shading interp;
            set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
            if i==1, ylabel('original'); end

            % Noisy digit
            subplot(2 + lpcs, ndig, i + ndig);
            pcolor(1:15, 16:-1:1, reshape(xt, 15, 16)'); shading interp;
            set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
            if i==1, ylabel('noisy'); end
        end

        % Denoised with current nb_pcs
        Xdt(i,:) = preimage_rbf(Xtr, best_sig2, Ud, xt, 'denoise');
        subplot(2 + lpcs, ndig, i + (2 + k - 1) * ndig);
        pcolor(1:15, 16:-1:1, reshape(Xdt(i,:), 15, 16)'); shading interp;
        set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
        if i==1, ylabel(['n=', num2str(nb_pcs)]); end
    end
end
