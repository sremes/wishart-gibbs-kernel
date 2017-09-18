function [u_new, B_new, Ku2, ZZ2, Kpq2] = lcm_interpolate(model, T_new)

% construct Ku_new and predict u_new
Kpq_new = make_Kpq(model.T, T_new, model.ell_u);
Kz_new = kron(gausskernel(model.T, T_new, model.ell_z), eye(model.Q));
Z_new = Kz_new' * (model.Kz \ model.Z);
Ku_new = (Z_new*model.Z'/model.Znu)' .* Kpq_new;
u_new = Ku_new' * (model.Ku \ model.u);

% Ku at the new points
Kpq2 = make_Kpq(T_new, T_new, model.ell_u);
ZZ2 = (Z_new*Z_new'/model.Znu);
Ku2 = ZZ2 .* Kpq2;

% predict B_new
KB_new = kron(gausskernel(model.T, T_new, model.ell_b, model.sigma_b), eye(model.Q));
Bm_new = KB_new' * (model.KB \ model.Bm);
B_new = zeros(model.M*length(T_new), model.Q*length(T_new)); % M*Q blocks on diagonal
idx = kron(eye(length(T_new)), ones(model.M, model.Q)) == 1;
B_new(idx) = reshape(Bm_new', [], 1);