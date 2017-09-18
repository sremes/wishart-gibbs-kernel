function [model] = up_f(model)
    model.f_mu = model.w' * model.u + model.b;
    model.f_mu(model.f_mu < -7) = -7; % cut-off extreme values
    model.f_mu(model.f_mu > 7) = 7;
    model.f_var = 1;
    for s = 1:model.S
        if model.y(s) > 0
            [model.f(s),model.ff(s),model.f_ent(s)] = mean_trunc_gaussian(model.f_mu(s), model.f_var, model.nu, inf);
        else
            [model.f(s),model.ff(s),model.f_ent(s)] = mean_trunc_gaussian(model.f_mu(s), model.f_var, -inf, -model.nu);
        end
        if ~isfinite(model.f(s))
            keyboard
        end
    end
end

function [m,m2,ent] = mean_trunc_gaussian(mu, sigma, a, b)
    alpha = (a-mu)/sigma;
    beta = (b-mu)/sigma;
    eps = 1e-14;
    Z = normcdf(beta) - normcdf(alpha);

    m = mu + (normpdf(alpha) - normpdf(beta)) / Z;
    
    if isinf(alpha)
        temp = -beta*normpdf(beta);
    elseif isinf(beta)
        temp = alpha*normpdf(alpha);
    else 
        temp = alpha*normpdf(alpha) - beta*normpdf(beta);
    end
    
    mvar = 1 + (temp)/Z - ((normpdf(alpha) - normpdf(beta)) / Z)^2;
    m2 = m^2 + mvar;
    
    ent = log(Z) + temp/(2*Z);
end
