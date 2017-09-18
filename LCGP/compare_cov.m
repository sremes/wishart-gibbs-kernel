function [score,sigma_best,perm_best] = compare_cov(sigma_true, sigma_est, N)
%% Compare two covariance matrices that may have permuted variables
% i.e. we need to maximize the score w.r.t. the permutation matrices.

% generate all permutations to loop through
D = size(sigma_true,1);
Q = D / N;
all_perms = perms(1:Q);
score = -1;
for k = 1:size(all_perms,1)
    P = eye(Q);
    P = P(all_perms(k,:),:);
    P = kron(eye(N), P);
    sigma_perm = P*sigma_est*P';
    tmp = corr(sigma_true(:), sigma_perm(:));
    if tmp > score
        score = tmp;
        sigma_best = sigma_perm;
        perm_best = all_perms(k,:);
    end
end
