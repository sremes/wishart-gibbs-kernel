function [model] = up_tau(model)

tau0 = 1e-6;
model.tau0 = tau0;

model.tau_a = tau0 + 0.5*numel(model.x);
res1 = model.x_vec.^2;
res2 = -2*model.u.*(model.B'*model.x_vec);
res3 = model.BB.*model.uu;
model.tau_b = tau0 + 0.5*(sum(res1(:))+sum(res2(:))+sum(res3(:)));

model.tau = model.tau_a / model.tau_b;
model.logtau = psi(model.tau_a) - log(model.tau_b);