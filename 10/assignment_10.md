## Logistric Regression using non-linear feature functions with a quadratic regularization for a binary classification 

### 1. Problem definition

#### (1). Training Data

The training data consists of $`\{(x_i, y_i, \ell_i)\}_{i=1}^n\}`$ where $`(x_i, y_i) \in \mathbb{R}^2`$ represent a point in 2-dimensional coordinate and $`\ell_i \in \{0, 1\}`$ represents its class label.

#### (2). Testing Data

The testing is used to evaluate the estimation of model and improve its generality. The testing data also consists of $`\{(x_i, y_i, \ell_i)\}_{i=1}^n\}`$ where $`(x_i, y_i) \in \mathbb{R}^2`$ represent a point in 2-dimensional coordinate and $`\ell_i \in \{0, 1\}`$ represents its class label.

#### (3). Linear regression function with a quadratic regularization

Let $`p = (x, y)`$ be a point and $`\theta = (\theta_0, \theta_1, \cdots, \theta_k)`$ be a set of model parameters. The regression function is defined by the linear combination of model parameters and a set of feature functions $`f_k \colon \mathbb{R}^2 \mapsto \mathbb{R}`$ as follows:

```math
f(x, y; \theta) = \theta_0 \cdot f_0(x, y) + \theta_1 \cdot f_1(x, y) + \cdots + \theta_k \cdot f_k(x, y)
```

where $`\theta \in \mathbb{R}^{k+1}`$ and $`(f_0, f_1, \cdots, f_k) \in \mathbb{R}^{k+1}`$.

#### (4). Sigmoid function

The sigmoid function is defined by:

```math
\sigma(z) = \frac{1}{1 + \exp{(-z)}}
```

The derivative of sigmoid function is defined by:

```math
\sigma'(z) = \sigma(z) (1 - \sigma(z)).
```

#### (5). Logistic regression function

The logistic regression function is defined by:

```math
h(x, y ; \theta) = \sigma(f(x, y ; \theta)).
```

#### (6). Objective function

The objective function for the binary classification based on the logistic regression function with a quadratic regularization is defined by:

```math
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \{ - \ell_i \log{(h_i)} - (1 - \ell_i) \log{(1 - h_i)} \} + \frac{\alpha}{2} \| \theta \|_2^2
```

where $`h_i = \sigma(f(x_i, y_i ; \theta))`$ and

```math
\| \theta \|_2^2 = \theta_0^2 + \theta_1^2 + \cdots + \theta_k^2
```

#### (7). Optimization using the gradient descent algorithm

The optimization of the objective function $`\mathcal{L}(\theta)`$ with respect to the model parameters $`\theta`$ using the gradient descent algorithm is given by:
```math
\theta^{t+1} \coloneqq \theta^{t} - \eta \nabla \mathcal{L}(\theta)  
```
where $`t`$ denotes iteration and $`\eta`$ denotes learning rate.

#### (8). Optimal classifier

The classifier for point $`(x, y)`$ is given by the logistic regression function with obtained model parameters $`\theta^*`$ as given by:
```math
\hat{h}(x, y ; \theta^*)
```
where 
```math
\theta^* = \arg\min_\theta \mathcal{L}(\theta)
```

### 2. Complete the notebook 

- download the notebook file [assignment_10.ipynb](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/10/assignment_10.ipynb) 
- download the data files [assignment_10_data_train.csv](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/10/assignment_10_data_train.csv) for training and [assignment_10_data_test.csv](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/10/assignment_10_data_test.csv) for testing
- complete the `codes section` in the notebook so that the results in the `results section` in the notebook can be produced as expected
 
### 3. list of submission

- completed notebook file in PDF format
- github history in the course of completiing the notebook file in PDF format
- it is recommended to make `git commit` and `git push` as many as possible (e.g. in every 10 minutes)
