function [model] = up_kernel(model)
%% Update K_u = ZZ^T \circ K'
% Elements z_ij(t) ~ GP(0, K_z)
% Vectorized into matrix Z = [z(1); ...; z(N)], with each z(t) a Q*nu matrix

[~,nu] = size(model.Z);
if model.optim.whitening
%     e = 1e-2;
%     d = checkgrad(@grad_whiten, {model.Lz\model.Z, log(model.omega), log(model.ell_u)}, e, model);
    [theta,~] = minimize_v2({model.Lz\model.Z, log(model.omega), log(model.ell_u)}, @grad_whiten, -20, model);
    Z = model.Lz*theta{1};
else
    [theta,~] = minimize_v2({model.Z, log(model.omega), log(model.ell_u)}, @gradient, -20, model);
    Z = theta{1};
end
model.omega = exp(theta{2});
model.ell_u = exp(theta{3});
model.Kpq = make_Kpq(model.T, model.T, model.ell_u);
model.Ku = Z*Z'/nu .* model.Kpq + model.omega * eye(size(model.Kpq));
model.Ku_inv = model.Ku \ eye(size(model.Ku));
model.Z = Z;
end

function [ftheta,dtheta] = gradient(theta, model)
    [NQ,nu] = size(model.Z);
    Z = theta{1};
    lomega = theta{2};
    lell_u = theta{3};
    Kpq = make_Kpq(model.T,model.T,exp(lell_u));
    Ku = Z*Z'/nu .* Kpq + exp(lomega) * eye(size(model.Kpq));
    Ku_inv = Ku \ eye(size(Ku));
    lb = -model.S/2 * logdet(Ku); 
    lb = lb -0.5 * sum(sum(Ku_inv .* model.uu)); % quadratic term
    lb = lb + sum(logmvnpdf(Z', zeros(size(Z')), model.Kz)); % gp prior of Z
    ftheta = -lb;  % negative lower bound to minimize
    
    Lambda = Ku_inv * model.uu * Ku_inv - model.S * Ku_inv;
    Lambda = Lambda(:);

    dZ = - model.Kz_inv * Z; 
    parfor i = 1:NQ
        for j = 1:nu
            dz = sparse(i,1:NQ,Z(:,j),NQ,NQ) + sparse(1:NQ,i,Z(:,j),NQ,NQ);
            nablaCq = dz/nu .* Kpq;
            dZ(i,j) = dZ(i,j) + 0.5 * sum(nablaCq(:) .* Lambda);
        end
    end
    dell = zeros(model.Q,1);
    T = model.T;
    ZZ = Z*Z'/nu;
    parfor k = 1:model.Q
        dK = ZZ .* grad_Kpq(T, T, exp(lell_u), k);
        dell(k) = 0.5*sum(sum(Lambda .* dK(:)));
    end
    domega = 0.5*sum(diag(reshape(Lambda, size(Ku_inv)))); % trace of Lambda
    dtheta = {-dZ, -(exp(lomega)*domega), -(2*exp(lell_u).*dell)}; % negative lower bound to minimize
end

function [fthetaw,dthetaw] = grad_whiten(thetaw, model)
    [fthetaw,dtheta] = gradient({model.Lz*thetaw{1}, thetaw{2}, thetaw{3}}, model);
    dthetaw = {model.Lz'*dtheta{1}, dtheta{2}, dtheta{3}};
end
