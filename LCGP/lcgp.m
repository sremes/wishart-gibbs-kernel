function [model] = lcgp(T, x, y, ell_u, ell_b, ell_z, opts)
%% Implements the LCGP model.
% T: input time points (or generic multivariate input N×D)
% x: data
% y: class labels
% ell_u: Q latent variable length-scales (determines the number of latents)
% ell_b: mixing length-scale
% ell_z: latent covariance length-scale
% opts: extra options

% check some options
def_opts.iters = 50;
def_opts.optimize_omega = true;
def_opts.omega = 1e-3;
def_opts.use_ard = false;
def_opts.sigma_b = 1;
def_opts.nu = 0;
def_opts.latent_correlation = true;
def_opts.sparse_mixing = false;

if nargin > 6
    opts = parse_opts(def_opts, opts);
else
    opts = def_opts;
end

model.opts = opts;

model.optim.whitening = true; % use whitened gradient updates for Z

% data size
[model.N,model.M,model.S] = size(x);
model.x = x;
model.x_vec = reshape(permute(x, [2 1 3]), [model.M*model.N, model.S]); % read data in M-size blocks
model.y = (y - min(y)) / (max(y) - min(y)); % scale to {0,1}
model.y = 2*model.y - 1; % scale to {-1,1}
model.Q = length(ell_u);
model.T = T;
model.nu = opts.nu;

% compute kernels and choleskys
model.ell_u = ell_u;
model.omega = opts.omega;

% latent correlation kernel Ku = ZZ' .* Kpq + omega*I
model.ell_z = ell_z;
model.Znu = model.Q; % "full-rank" Z
model.Kpq = make_Kpq(model.T, model.T, ell_u);

model.Kz = kron(gausskernel(model.T, model.T, ell_z, 1, model.omega), eye(model.Q));
model.Lz = chol(model.Kz,'lower');
model.Kz_inv = model.Kz \ eye(size(model.Kz));
model.Z = kron(ones(model.N,1), eye(model.Q, model.Znu));
model.Z = model.Z + 0.001*randn(size(model.Z));

model.Ku = model.Z*model.Z' .* model.Kpq + model.omega * eye(size(model.Kpq));
model.Ku_inv = model.Ku \ eye(size(model.Ku));


% kernel for B_m
model.ell_b = ell_b;
model.sigma_b = opts.sigma_b;
model.KB = gausskernel(model.T,model.T,ell_b,model.sigma_b,.001);
model.KB_inv = inv(model.KB);

% init variables
model.B = kron(eye(model.N), model.sigma_b*randn(model.M,model.Q));
model.Bm = reshape(model.B(model.B(:) ~= 0),[model.Q*model.N model.M]);
model.BB = model.B' * model.B;
model.tau = 1; % noise precision

% classifier weights and auxiliary variable
if model.S > 1 && ~isempty(model.y)
    model.lambda_b = 1;
    if opts.use_ard
        model.lambda_w = ones(model.Q*model.N,1);
    else
        model.lambda_w = 1;
    end

    model.w = randn(model.N*model.Q,1) ./ sqrt(model.lambda_w);
    model.ww = model.w * model.w';
    model.b = randn / sqrt(model.lambda_b);

    model.f = model.y;
end

% run the VB inference
for iter = 1:opts.iters
    fprintf('Iteration %d\n', iter);
    model = up_u(model);
    if opts.latent_correlation && iter > 1
        model = up_kernel(model);
    end
    if model.S > 1  && ~isempty(model.y) % do we do classification
        model = up_f(model);
        if opts.use_ard
            model = up_w_b(model);
        else
            model = up_w_b_ridge(model);
        end
    end     
    model = up_B(model,opts);
    model = up_tau(model);
    
    model.lb(iter) = lower_bound(model);
    fprintf('VB lower bound: %f\n', model.lb(iter));
end

end

function parsed_opts = parse_opts(default_opts, in_opts)
    in_fields = fieldnames(in_opts);
    n_in_fields = numel(in_fields);

    parsed_opts = default_opts;

    for i_field = 1:numel(in_fields)
        cur_field = in_fields{i_field};

        if isfield(default_opts, cur_field)
            parsed_opts.(cur_field) = in_opts.(cur_field);
        else
            error('parse_opts:input', 'Unknown field name `%s`', cur_field);
        end
    end
end
