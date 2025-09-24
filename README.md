# SVM-analysis
MATLAB implementations of Support Vector Machines (SVMs) for the following tasks: classification, regression, time series prediction, unsupervised learning, and large-scale problems. Covers LS-SVM, kernel PCA, ARD, robust regression, spectral clustering, and fixed-size SVMs.

## Assignment 1 – Classification and Regression  
Focus: binary classification and function approximation using LS-SVMs.  

- **Classification tasks**:  
  - Linear vs nonlinear kernels  
  - Hyperparameter tuning (γ, σ²) via cross-validation and Bayesian framework  
  - Multi-class extensions  
- **Regression tasks**:  
  - LS-SVM regression with RBF kernels  
  - Model comparison to least squares  
  - Visualization of decision boundaries and regression fits  

---

## Assignment 2 – Function Estimation and Time Series Prediction  
Focus: regression, robustness, feature selection, and temporal modeling.  

- **Function estimation**:  
  - Sinc function regression with LS-SVMs  
  - Hyperparameter tuning (grid search, simplex, Bayesian framework)  
  - Automatic Relevance Determination (ARD) for feature selection  
  - Robust regression with outlier handling  
- **Time series prediction**:  
  - **Logmap dataset**  
  - **Santa Fe laser dataset**  
  - Linear vs nonlinear autoregressive models (AR/NAR)  
  - Parameter tuning: model order, γ, σ²  

---

## Assignment 3 – Unsupervised Learning and Large-Scale Problems  
Focus: kernel-based dimensionality reduction and scalable SVMs.  

- **Kernel PCA**:  
  - Nonlinear feature extraction and denoising (yin-yang toy dataset)  
  - Comparison to linear PCA  
- **Spectral clustering** *(optional)*:  
  - Kernel similarity matrices and Laplacian-based clustering  
- **Fixed-size LS-SVM**:  
  - Nyström approximation and ℓ₀ penalties for sparse solutions  
  - Applications:  
    - **Digit denoising (KPCA)** – handwritten numeral dataset  
    - **Shuttle dataset (classification)** – [UCI Statlog Shuttle](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle))  
    - **California Housing dataset (regression)** – [UCI California Housing](http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)  

---

## Requirements  
- MATLAB (R2022a or later recommended)  
- **Toolboxes**:  
  - [LS-SVMlab](https://www.esat.kuleuven.be/sista/lssvmlab/)  
  - SVM and fixed-size LS-SVM scripts  

---

## References  
- J.A.K. Suykens, T. Van Gestel, J. De Brabanter, B. De Moor, J. Vandewalle, *Least Squares Support Vector Machines*, World Scientific, 2002.   
