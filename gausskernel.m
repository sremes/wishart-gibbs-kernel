function [K,dK] = gausskernel(X1,X2,hyp)
    omega = 1e-6;
    ell = exp(hyp.log_ell);
    sigma = exp(hyp.log_sigma);

    R = pdist2(X1,X2);
	if length(X1) == length(X2)
		K = sigma^2 * exp(-0.5* R.^2 / ell^2) + omega^2*eye(size(X1,1));
	else
		K = sigma^2 * exp(-0.5* R.^2 / ell^2);
	end

    dK.log_ell = 2*R.*K / ell^2;
    dK.log_sigma = 2*K;
end

