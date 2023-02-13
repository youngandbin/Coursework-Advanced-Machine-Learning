# Coursework-Advanced-Machine-Learning

### 1. Image Classification

- Description
  - build models from scratch
    - KNN, SVM, DT, RF 
  - experiment on CIFAR dataset
    - accuracies and runtime for both tarning and testing
      - w.r.t. a varying number of train data points
    - hyperparameter search with CV
  - data
    - CIFAR-10, CIFAR-100
  - advanced
    - comparisons with other data (Fashion MNIST, Caltech-256) and models (CNN, ResNet101)
    - ablation studies
- Code
  - ML_models.ipynb
    - KNN, SVM, Decision tree, Random forest

### 2. Gaussian Process Regression

- Description
  - build models from scratch
    - linear and nonlinear GP regression (vector-valued model)
      - i.i.d. Gaussian noise model 
      - isotropic Gaussian kernel (for nonlinear)
  - experiment on SARCOS dataset
    - accuracies and runtime 
    - accuracy surface w.r.t. varying hyperparameters
  - advanced
    - computationally efficient GP regression using the subset of regressors (SOR) approximation 
    - comparisons with other regression algorithms
    - other kernels
- Code

### 3. Neural Processes

- A survey paper on Neural Processes
- Includes the following papers;
  - Neural Processes(NPs)
  - Conditional Neural Processes(CNPs)
  - Attentive Neural Processes(ANPs)
  - Sequential Neural Processes(SNPs)
  - The Functional Neural Process(FNPs)

### 4. Neural Network Regression

- Description
  - compare models
    - RF, GP, MLP, CNN, and more
  - data
    - 3D Hand Pose Estimation 
