# Logistic Regression from Scratch

An educational machine learning project implementing **Logistic Regression from scratch using Python and NumPy**.

The project focuses on understanding how logistic regression works internally, rather than relying on a high-level machine learning library for the model itself.

## Overview

This project covers the main steps involved in building a logistic regression model:

* Sigmoid function
* Cost function
* Gradient calculation
* Gradient Descent
* Binary classification
* Model prediction
* L2 Regularization
* Model evaluation

The implementation uses **NumPy** for numerical computations and **Matplotlib** for data visualization.

## Technologies

* Python
* NumPy
* Matplotlib

## Project Structure

```text
logistic-regression-from-scratch/
│
├── LogisticRegression.py    # Main implementation
├── utils.py                 # Helper and visualization functions
├── public_tests.py          # Educational tests
├── test_utils.py            # Test utilities
├── data/                    # Training datasets
├── images/                  # Visualization results
└── README.md
```

## What I Learned

Through this project, I practiced implementing the core components of logistic regression manually and gained a better understanding of:

* How the sigmoid function converts model output into probabilities
* How the logistic regression cost function is calculated
* How gradients are computed
* How Gradient Descent updates model parameters
* How regularization helps control model complexity
* How a classification model makes predictions from input features

## Educational Purpose

This is an **educational project** created to practice and understand the concepts behind Logistic Regression and Gradient Descent.

The implementation was developed while studying machine learning concepts and completing practical exercises.

## Example

The model can be trained using Gradient Descent and then used to predict binary outcomes based on input features.

Example workflow:

```python
w, b = gradient_descent(
    X,
    y,
    w_init,
    b_init,
    cost_function,
    gradient_function,
    alpha,
    num_iters
)

predictions = predict(X, w, b)
```

## Results

The implementation was tested on binary classification datasets and achieved approximately **92% training accuracy** on one of the classification tasks.

## Future Improvements

Some possible improvements for this project include:

* Creating a cleaner reusable Logistic Regression class
* Adding more evaluation metrics
* Supporting multiple datasets
* Adding a standalone training script
* Improving visualizations and experiment tracking

## Author

**Mohammad Motiei**

Computer Engineering Student | Software & Machine Learning Enthusiast
