type = 'c';    % classification
gam = 1;       % regularization parameter
t = 1;         % offset for polynomial kernel
degrees = 1:5; % degrees to test

for i = 1:length(degrees)
    degree = degrees(i);
    
    disp(['Polynomial kernel of degree ', num2str(degree)]);

    % Train the LS-SVM
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam, [t; degree], 'poly_kernel'});

    % Plot the decision boundary
    figure;
    plotlssvm({Xtrain, Ytrain, type, gam, [t; degree], 'poly_kernel', 'preprocess'}, {alpha, b});
    title(['LS-SVM Decision Boundary - Polynomial Degree ', num2str(degree)]);

    % Predict on the test set
    [Yht, Zt] = simlssvm({Xtrain, Ytrain, type, gam, [t; degree], 'poly_kernel'}, {alpha, b}, Xtest);

    % Compute test error
    err = sum(Yht ~= Ytest);
    fprintf('\nOn test set (degree %d): #misclassified = %d, error rate = %.2f%%\n', degree, err, err/length(Ytest)*100);

    disp('Press any key to continue to next degree...');
    pause;
end