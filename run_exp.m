%% Run a simple experiment comparing Hadamard vs. Kronecker kernel

rng(2);
% Make data
D = 3;
N = 200;
R = D; % rank
M = 500; % test points

model.T = linspace(-1,1,N)';
T_test = linspace(-1.1,1.1,M)';
model.ell_u = [0.05, 0.05, 0.05]';
model.Kpq = make_Kpq(model.T, model.T, model.ell_u);
model.Q = D;
% Z
hyp_z.log_ell = log(0.25); hyp_z.log_sigma = log(1);
model.ell_z = 0.25;
model.Kz = kron(gausskernel(model.T, model.T, hyp_z), eye(D));
model.Kz_inv = inv(model.Kz);
model.Lz = chol(model.Kz, 'lower');
% construct a changing correlation between the outputs
z0 = eye(R) + diag([0.99 -0.99],1) + diag([0.99 -0.99],-1);
z1 = eye(R) + diag([-0.99 0.99],1) + diag([-0.99 0.99],-1);
g = 1-exp(-100*model.T.^2);
model.Z = [kron(g(1:end/2), z0); kron(g(end/2+1:end), z1)];
% Ku and u
model.omega = 2^2;
model.Ku = model.Z * model.Z' .* model.Kpq  + 1e-3 * eye(size(model.Kpq));
u_clean = chol(model.Ku)' * randn(size(model.Ku,1),1); % sample data
model.Ku = model.Ku + model.omega * eye(size(model.Kpq));
model.u = u_clean + sqrt(model.omega) * randn(size(u_clean));
model.uu = model.u * model.u';

u = reshape(model.u, [D,N])';
u_clean = reshape(u_clean, [D,N])';
figure(1); plot(model.T, u_clean, 'LineWidth', 3);

%% Run Kronecker
f_kron_best = Inf; hyp_kron_best = [];
for iter = 1:100
    hyp_kron.log_sigma = 0; hyp_kron.log_ell = log(rand); 
    hyp_kron.z = cellfun(@(x) randn(D,1), cell(D,1), 'UniformOutput', false);
    hyp_kron.log_noise = log(1);
    [hyp_kron, f_kron] = minimize_v2(hyp_kron, @nlogp_kronecker, -100, model.u, {model.T, (1:D)'});
    if f_kron(end) < f_kron_best
        f_kron_best = f_kron(end);
        hyp_kron_best = hyp_kron;
    end
end
%
Kreg = coreg_kernel(model.T, model.T, hyp_kron_best);
Kgauss = gausskernel(model.T, model.T, hyp_kron_best);
K_kron = kron(Kgauss, Kreg); % + exp(2*hyp_kron_best.log_noise)*eye(D*N);
figure(2); imagesc(K_kron); colorbar; title('Kronecker kernel')

%% Run Hadamard
lb_best = Inf; model_best = [];
for iter = 1:100
    model_init = model; % init model randomly
    model_init.Z = kron(ones(N,1), eye(R) + 0.001*randn(R)); 
    model_init.ell_u = rand(D,1); model_init.omega = 1;
    model_had = optim_hadamard(model_init);
    if model_had.NLL(end) < lb_best
        model_best = model_had;
        lb_best = model_had.NLL(end);
    end
end
model_had = model_best;

figure(3); 
K_hadam = model_had.Ku - model_had.omega*eye(size(model_had.Ku));
imagesc(K_hadam); colorbar; title('Hadamard kernel')

%% Plot predictions Kronecker
K_kron_xx = kron(gausskernel(model.T, model.T, hyp_kron_best), Kreg);
K_kron_xz = kron(gausskernel(model.T, T_test, hyp_kron_best), Kreg);
K_kron_zz = kron(gausskernel(T_test, T_test, hyp_kron_best), Kreg);
K_kron = K_kron_xx + exp(2*hyp_kron_best.log_noise)*eye(D*N);
u_test = reshape(K_kron_xz' * (K_kron \ model.u), [D M])';
u_test_std = sqrt(diag(K_kron_zz - K_kron_xz'*(K_kron_xx\K_kron_xz)) + exp(2*hyp_kron_best.log_noise));
u_test_std = reshape(u_test_std, [D M])';
figure(4); clf; 
plot(T_test, u_test, 'LineWidth', 3); 
hold on; set(gca, 'ColorOrderIndex', 1)
plot(model.T, u, '.')
set(gca, 'ColorOrderIndex', 1)
plot(model.T, u_clean, '--','LineWidth',2)
title('Kronecker predictions')
xlim([-1.1 1.1])

% predictive log likelihood


%% Plot predictions Hadamard
[u_test, Ku2] = predict_hadamard(model_had, T_test);
u_test = reshape(u_test, [D M])';
u_test_std = sqrt(reshape(diag(Ku2), [D M])' + model_had.omega);
figure(5); clf; 
plot(T_test, u_test, 'LineWidth', 3); 
hold on; set(gca, 'ColorOrderIndex', 1)
plot(model.T, u, '.')
set(gca, 'ColorOrderIndex', 1)
plot(model.T, u_clean, '--','LineWidth',2)
title('Hadamard predictions')
xlim([-1.1 1.1])

%% Errors between the found posterior and the true function
post_kron = K_kron_xx*(K_kron\model.u);
mse_kron = mean((post_kron-reshape(u_clean',[],1)).^2)

post_had = predict_hadamard(model_had, model.T);
mse_had = mean((post_had-reshape(u_clean',[],1)).^2)

mse_null = mean(u_clean(:).^2)
