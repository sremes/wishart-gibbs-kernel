function [lb] = lower_bound(model)
%% Compute the variational lower bound.
% Can be used e.g. to monitor convergence of the algorithm.

% likelihood term
res1 = model.x_vec.^2;
res2 = -2*model.u.*(model.B'*model.x_vec);
res3 = model.BB.*model.uu;
lb = model.N*model.M*model.S*model.logtau/2 - ... 
    0.5*model.tau*(sum(res1(:))+sum(res2(:))+sum(res3(:)));

% u and Z
lb = lb - model.S/2 * logdet(model.Ku); % logdet
lb = lb - 0.5 * sum(sum(model.Ku_inv .* model.uu)); % quadratic term
lb = lb + sum(logmvnpdf(model.Z', zeros(size(model.Z')), model.Kz)); % gp prior of Z
lb = lb + 0.5*model.S*logdet(model.u_sigma); % entropy

% B
lb = lb - 0.5*sum(sum((model.Bm*model.Bm'+model.M*model.B_sigma).*model.KB_inv));
lb = lb + 0.5*model.M*logdet(model.B_sigma); % entropy

% tau
lb = lb + (model.tau0-1)*model.logtau - model.tau0*model.tau;
lb = lb + (model.tau_a - log(model.tau_b) + gammaln(model.tau_a) - (model.tau_a-1)*psi(model.tau_a)); % entropy

%fprintf('DEBUG: lb after tau: %f\n', lb);
if model.S > 1 && ~isempty(model.y)
    % w and b
    if length(model.lambda_w) > 1
        diag_term = diag([model.lambda_w; model.lambda_b]);
    else
        diag_term = diag([model.lambda_w*ones(length(model.w),1); model.lambda_b]);
    end
    lb = lb - 0.5*sum(sum(model.wwbb .* diag_term)) + ...
       sum(model.loglambda_w * length(model.w)/numel(model.loglambda_w))/2 + model.loglambda_b/2;
    lb = lb + 0.5*logdet(model.wb_sigma); % entropy

    % lambda_{w,b}
    lb = lb + sum((model.lambda0-1)*model.loglambda_w - model.lambda0*model.lambda_w);
    lb = lb + sum((model.lambda_w_alpha - log(model.lambda_w_beta) + ...
        gammaln(model.lambda_w_alpha) - (model.lambda_w_alpha-1)*psi(model.lambda_w_alpha))); %entropy
    lb = lb + sum((model.lambda0-1)*model.loglambda_b - model.lambda0*model.lambda_b);
    lb = lb + sum((model.lambda_b_alpha - log(model.lambda_b_beta) + ...
        gammaln(model.lambda_b_alpha) - (model.lambda_b_alpha-1)*psi(model.lambda_b_alpha))); %entropy

    % f
    lb = lb - 0.5*(sum(model.ff) + sum(sum(model.uu .* model.ww)) + model.S*model.bb ...
        - 2*model.f'*model.u'*model.w - 2*model.b*sum(model.f) + 2*sum(model.wb'*model.u));
    lb = lb + sum(model.f_ent); %entropy
end
