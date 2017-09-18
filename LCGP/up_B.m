function [model] = up_B(model,opts)

Sigma_B_inv = model.KB_inv + model.tau * model.uu_tilde;
model.B_sigma = Sigma_B_inv \ eye(size(Sigma_B_inv)); % shared for all B_m

% Update B
B = zeros(model.N*model.Q, model.M);
for s = 1:model.S
    B = B + model.u_tilde(:,:,s)' * model.x(:,:,s) * model.tau;
end
B = model.B_sigma * B;
model.Bm = B;

% Construct block-diagonal B
model.B = zeros(model.M*model.N, model.Q*model.N); % M*Q blocks on diagonal
idx = kron(eye(model.N), ones(model.M, model.Q)) == 1;
model.B(idx) = reshape(model.Bm', [], 1);

% %model.BB = model.B' * model.B + model.B_sigma;
sigma = model.B_sigma;
idx = kron(eye(model.N), ones(model.Q));
sigma(idx == 0) = 0;
model.BB = model.B' * model.B + model.M*sigma; % Q*Q blocks

