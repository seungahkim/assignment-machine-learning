## Linear Regression

### 1. Probelm Definition 

#### (1). Formulation of the linear regression 

- data

```math
\{ (x_i, y_i, z_i) \}_{i=1}^n = \{ (x_1, y_1, z_1), (x_2, y_2, z_2), \cdots, (x_n, y_n, z_n) \}, \quad (x_i, y_i, z_i) \in \mathbb{R}^3, \forall i
```

- model 

```math
\hat{f}(\theta ; x, y) = \theta_0 + \theta_1 x + \theta_2 y, \quad (\theta_0, \theta_1, \theta_2) \in \mathbb{R}^3
```

- model parameters

```math
\theta = (\theta_0, \theta_1, \theta_2)
```

- residual 

```math
\gamma_{i}(\theta) = z_i - \hat{f}(\theta ; x_i, y_i)
```

- objective function

```math
\mathcal{L}(\theta) = \frac{1}{2 n} \sum_{i=1}^n \gamma_{i}^2(\theta) = \frac{1}{2 n} \sum_{i=1}^n (\hat{f}(\theta ; x_i, y_i) - z_i)^2 = \frac{1}{2 n} \sum_{i=1}^n (\theta_0 + \theta_1 x_i + \theta_2 y_i - z_i)^2 
```

- solution

```math
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
```

#### (2). Optimization by Gradient Descent

- iterative optimization with gradient descent for the model parameters

```math
\theta^{(t+1)} \coloneqq \theta^{(t)} - \eta \, \nabla \mathcal{L}(\theta^{(t)})
```
where $`\eta > 0`$ in $`\mathbb{R}`$ is called learning rate.

- gradient descent step for the model parameter vector $`\theta = (\theta_0, \theta_1, \theta_2)`$

#### (3). Regression Surface

- three dimensional surface of regression function $`\hat{f}`$ with optimal model parameters $`\theta^*`$ 

```math
(x, y, \hat{f}(\theta^*, x, y))
```

### 2. Complete the notebook 

- download the notebook file [assignment_07.ipynb](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/07/assignment_07.ipynb) 
- download the data file [assignment_07_data.csv](https://gitlab.com/cau-class/machine-learning/2021-1/assignment/-/blob/master/07/assignment_07_data.csv)
- complete the `codes section` in the notebook so that the results in the `results section` in the notebook can be produced as expected
- parameter selection
>>> 
\# 1. $`\theta_0^{(0)} = 0`$ for the initial condition of $`\theta_0`$

\# 2. $`\theta_1^{(0)} = 0`$ for the initial condition of $`\theta_1`$

\# 3. $`\theta_2^{(0)} = 0`$ for the initial condition of $`\theta_2`$

\# 4. $`\eta = 0.01`$ for the learning rate

\# 5. maximum iteration of the gradient descent is $`1,000`$
>>>
    
### 3. list of submission

1. completed notebook file in PDF format
2. github history in the course of completiing the notebook file in PDF format
