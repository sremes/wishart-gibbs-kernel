function [model] = up_w_b(model)

one = ones(model.S,1);

Sigma_inv = [model.uu + diag(model.lambda_w), model.u * one;
             one' * model.u', model.S + model.lambda_b];
model.wb_sigma = Sigma_inv \ eye(size(Sigma_inv));

wb = model.wb_sigma * [model.u * model.f; one' * model.f];

model.w = wb(1:end-1); % <w>
model.ww = model.w * model.w' + model.wb_sigma(1:end-1, 1:end-1); % <w*w'>
model.b = wb(end); % <b>
model.bb = model.b^2 + model.wb_sigma(end,end);

model.wwbb = [model.w;model.b]*[model.w;model.b]' + model.wb_sigma;
model.wb = model.w*model.b + model.wb_sigma(1:end-1,end);

% update lambda_w,b
a0 = 1e-6;
model.lambda0 = a0;
model.lambda_w_alpha = a0 + 0.5;
model.lambda_w_beta = a0 + 0.5*(model.w.^2 + diag(model.wb_sigma(1:end-1,1:end-1)));
model.lambda_w = model.lambda_w_alpha ./ model.lambda_w_beta;
model.loglambda_w = psi(model.lambda_w_alpha) - log(model.lambda_w_beta);

model.lambda_b_alpha = a0 + 0.5;
model.lambda_b_beta = a0 + 0.5*(model.b^2 + model.wb_sigma(end,end));
model.lambda_b = model.lambda_b_alpha / model.lambda_b_beta;
model.loglambda_b = psi(model.lambda_b_alpha) - log(model.lambda_b_beta);