function [model] = up_u(model)

if model.S > 1 && ~isempty(model.y)
    sigma_inv = model.Ku_inv + model.tau*model.BB + model.ww;
    model.u_sigma = sigma_inv \ eye(size(sigma_inv));
    model.u = model.u_sigma * (repmat((model.f - model.b)',[model.N*model.Q 1]).*repmat(model.w,[1 model.S]) + ...
                model.tau*model.B'*model.x_vec);
else
    sigma_inv = model.Ku_inv + model.tau*model.BB;
    model.u_sigma = sigma_inv \ eye(size(sigma_inv));
    model.u = model.u_sigma * (model.tau*model.B'*model.x_vec);
end
model.uu = model.u * model.u' + model.S*model.u_sigma;

model.uu_tilde = 0*model.u_sigma;
for s = 1:model.S
    % following shapes needed in updating B:

    % (1*Q block diagonal [u^s(1)', ...])
    tmp = kron(eye(model.N), ones(1, model.Q)) == 1;
    tmp2 = zeros(model.N, model.N*model.Q);
    tmp2(tmp) = model.u(:,s);
    model.u_tilde(:,:,s) = tmp2;
    % (Q*Q block diagonal [u^s(1)*u^s(1)', ...])
    sigma = model.u_sigma;
    tmp = kron(eye(model.N), ones(model.Q,model.Q));
    sigma(~tmp) = 0; % remove off-diagonal blocks
    model.uu_tilde = model.uu_tilde + model.u_tilde(:,:,s)' * model.u_tilde(:,:,s) + sigma; 
end
