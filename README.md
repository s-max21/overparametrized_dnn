# Deep Overparametrized Neural Network Comparison

This repository contains code for creating a deep overparametrized neural network (dnn) and comparing its performance to other regression models. The dnn consists of K fully connected neural networks with L layers and r neurons per layer and combines the truncated outputs of these K neural networks in a last layer. The models are evaluated based on their mean squared error (MSE) and mean absolute error (MAE).

## Models

The following models are included in the comparison:

- Deep Neural Network (DNN)
- Neural Network with one hidden layer (NN1)
- Neural Network with three hidden layers (NN2)
- Neural Network with six hidden layers (NN3)
- K-Nearest Neighbors (KNN) Regression
- Radial Basis Function (RBF) Interpolation
- Decision Tree Regression

## Usage

Each model is defined in its own Python file in the repository. The `__init__.py` file imports the necessary functions from these files.

To create and evaluate a model, use the `train_and_evaluate` function for that model. For example, to train and evaluate the different NNs, use the `train_and_evaluate_nn` function.

To tune the parameters of a neural network or the k-nearest neighbors, use the `parameter_tuning_nn` or the `parameter_tuning_knn` functions.

## Results

The performance of the models is compared in terms of their MSE and MAE. The results are presented in the `model_comparison.ipynb` notebook.

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request.
