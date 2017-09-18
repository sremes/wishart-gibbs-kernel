function [l,g,K] = nlogp_kronecker(hyp, u, x)
% marginal likelihood and gradients 
% using kronecker inference on a multidimensional grid
% x: cell array of length P containing the input points along all P axes
% u: P-dimensional array
% hyp: hyper parameters
% components


% compute kernels
noise = exp(2*hyp.log_noise);
K = cell(2,1); dK = cell(2,1);
[K{1},dK{1}] = gausskernel(x{1},x{1},hyp);
[K{2},dK{2}] = coreg_kernel(x{2},x{2},hyp);

% compute MLL = log N(vec(u)|0, K{1} x ... x K{P} + sigma^2 I)
% following notation of GPatt of Wilson (2014) / Saatchi (2011)
P = 2;
Q = cell(P,1); V = cell(P,1); Qt = Q; %Vinv = V;
eig_vals = 1;
for p = 1:P
    [Q{p}, V{p}] = eig(K{p} + 1e-4*eye(numel(x{p})));
    Qt{p} = Q{p}';
    assert(all(isreal(V{p})),'non-real eigen values');
    assert(all(isreal(Q{p})),'non-real eigen vectors');
    eig_vals = kron(eig_vals, diag(V{p}));
end
eig_vals = real(eig_vals + noise);

Kinv_u = kron_mv(Q, kron_mv(Qt,u(:)) ./ eig_vals);
l = 0.5 * (sum(log(eig_vals)) + u(:)'*Kinv_u(:));


% GRADIENTS (Saatci's Thesis)
if nargout > 1
    diag_QtKQs = cell(P,1);
    for p = 1:P % precompute
        diag_QtKQs{p} = diag(Qt{p} * K{p} * Q{p});
    end
    % log sigma
    d_kernel = K;
    d_diag = diag_QtKQs;
    d_kernel{1} = dK{1}.log_sigma;
    d_diag{1} = diag(Qt{1} * d_kernel{1} * Q{1});
    g.log_sigma = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
    % log ell
    d_kernel = K;
    d_diag = diag_QtKQs;
    d_kernel{1} = dK{1}.log_ell;
    d_diag{1} = diag(Qt{1} * d_kernel{1} * Q{1});
    g.log_ell = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
    % z
    g.z = cell(size(dK{2}.z));
    for r = 1:size(dK{2}.z, 1)
        for d = 1:size(dK{2}.z, 2)
            d_kernel = K;
            d_diag = diag_QtKQs;
            d_kernel{2} = dK{2}.z{r,d};
            d_diag{2} = diag(Qt{2} * d_kernel{2} * Q{2});
            g.z{r,d} = kron_deriv(d_kernel, d_diag, Kinv_u, eig_vals);
        end
    end
    %noise
    g.log_noise = 0.5*(-Kinv_u'*Kinv_u + sum(1./eig_vals)) * (2*noise);
end

function g = kron_deriv(d_kernel, d_diag, alpha, eig_vals)
kron_diag = 1;
for d = 1:length(d_kernel) %fliplr(1:length(d_kernel))
    kron_diag = kron(kron_diag, d_diag{d});
end
trace_term = sum(kron_diag ./ eig_vals);
norm_term = alpha' * kron_mv(d_kernel, alpha);
g = -0.5*norm_term + 0.5*trace_term;

