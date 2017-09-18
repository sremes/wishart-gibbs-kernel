function [u_new, Ku2, ZZ2, Kpq2] = lcm_interpolate(model, T_new)

% construct Ku_new and predict u_new
Kpq_new = make_Kpq(model.T, T_new, model.ell_u);
hypz.log_ell = log(model.ell_z); hypz.log_sigma = 0;
Kz_new = kron(gausskernel(model.T, T_new, hypz), eye(model.Q));
Z_new = Kz_new' * (model.Kz \ model.Z);
Ku_new = (Z_new*model.Z'/size(model.Z,2))' .* Kpq_new;
u_new = Ku_new' * (model.Ku \ model.u);

% Ku at the new points
Kpq2 = make_Kpq(T_new, T_new, model.ell_u);
ZZ2 = Z_new * Z_new' / size(model.Z,2);
Ku2 = ZZ2 .* Kpq2;

Ku2 = Ku2 - Ku_new'*(model.Ku\Ku_new);