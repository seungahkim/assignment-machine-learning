## Logistric Regression for a binary classification 

### 1. Problem definition

#### (1). Training Data

The training data consists of $`\{(x_i, y_i, \ell_i)\}_{i=1}^n\}`$ where $`(x_i, y_i) \in \mathbb{R}^2`$ represent a point in 2-dimensional coordinate and $`\ell_i \in \{0, 1\}`$ represents its class label.

#### (2). Linear regression function

Let $`p = (1, x, y)`$ be a point and $`\theta = (\theta_0, \theta_1, \theta_2)`$ be a set of model parameters. The linear regression function is defined by:
```math
f(x, y; \theta) = \theta_0 * 1 + \theta_1 * x + \theta_2 * y = \theta^{T} p
```
where $`\theta \in \mathbb{R}^3`$ and $`p \in \mathbb{R}^3`$.

#### (3). Sigmoid function

The sigmoid function is defined by:
```math
\sigma(z) = \frac{1}{1 + \exp{(-z)}}
```
The derivative of sigmoid function is defined by:
```math
\sigma'(z) = \sigma(z) (1 - \sigma(z)).
```

#### (4). Logistic regression function

The logistic regression function is defined by:
```math
h(x, y ; \theta) = \sigma(f(x, y ; \theta)).
```

#### (5). Objective function

The objective function for the binary classification based on the logistic regression function is defined by:
```math
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \{ - \ell_i \log{(h_i)} - (1 - \ell_i) \log{(1 - h_i)} \}
```
where $`h_i = \sigma(f(x_i, y_i ; \theta))`$.

#### (6). Optimization using the gradient descent algorithm

The optimization of the objective function $`\mathcal{L}(\theta)`$ with respect to the model parameters $`\theta`$ using the gradient descent algorithm is given by:
```math
\theta^{t+1} \coloneqq \theta^{t} - \eta \nabla \mathcal{L}(\theta)  
```
where $`t`$ denotes iteration and $`\eta`$ denotes learning rate.

#### (7). Optimal classifier

The classifier for point $`(x, y)`$ is given by the logistic regression function with obtained model parameters $`\theta^*`$ as given by:
```math
\hat{h}(x, y ; \theta^*)
```
where 
```math
\theta^* = \arg\min_\theta \mathcal{L}(\theta)
```

### 2. Complete the notebook 

- download the notebook file [assignment_08.ipynb](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/08/assignment_08.ipynb) 
- download the data file [assignment_08_data.csv](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/08/assignment_08_data.csv)
- complete the `codes section` in the notebook so that the results in the `results section` in the notebook can be produced as expected
 
### 3. list of submission

- completed notebook file in PDF format
- github history in the course of completiing the notebook file in PDF format
