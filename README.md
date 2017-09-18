# Wishart-Gibbs Kernel for Latent Variable Couplings and Multi-Task Learning

## Method

This code implements the method presented in the following paper:

 * Remes, Heinonen, Kaski (2017). 'A Mutually-Dependent Hadamard Kernel for Modelling Latent Variable Couplings'. Accepted for ACML 2017. Preprint: https://arxiv.org/abs/1702.08402

The proposed Wishart-Gibbs kernel can be used both in multi-task/output Gaussian Processes and in modelling latent variable couplings in latent factor models such as the Gaussian process regression network (GPRN). In the paper we propose a model called LCGP, which finds a latent space that jointly produces multiple real outputs through input-dependent mixing matrices and a classification output through a Probit link function.

## Example

Here, we show a simple application for the proposed kernel in multi-output GP. We generate a toy dataset where the coupling between outputs changes clearly.

```matlab
% Put data into model struct:
model.T = T; % inputs
model.u = u; % outputs

% Initialize parameters with Q output variables, N samples:
model.Q = 3;
model.ell_u = rand(model.Q,1); % length-scales for the Gibbs kernel
model.ell_z = ell_z;
model.Kz = kron(gausskernel(T, T, ell_z), eye(model.Q)); % GP kernel for Wishart variables Z
model.Kz_inv = inv(model.Kz); model.Lz = chol(model.Kz, 'lower'); % pre-compute these
model.Z = kron(ones(N,1), eye(Q)); % init with a diagonal covariance
model.omega = 1; % noise variance

% Optimize model (Z, ell_u and log_noise):
model = optim_hadamard(model); 
```

See `run_exp.m` for a full code to run this example, that also runs a comparison using a Kronecker kernel.

## LCGP

For LCGP we provide a function 'lcgp.m' that initializes the Wishart-Gibbs kernel used for the latent variables, as well as all the variational distributions, and runs the variational inference algorithm.

```matlab
model = lcgp(T, x, y, ell_u, ell_b, ell_z, opts);
```

Here `T` includes the inputs (*N*x*D* matrix of *D*-dimensional inputs of length *N*). Data matrix `x` is of size *N*x*M*x*S*, where *M* is the output dimensionality and *S* the number samples. Class labels are given in vector `y` of length *S*. Gibbs kernel length-scales are given in vector `ell_u` of length *Q* for each of the latent signals. Length-scales for the mixing matrix and Wishart variables are given in `ell_b` and `ell_z`.
