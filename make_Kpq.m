function Kpq = make_Kpq(x1, x2, ell)

Q = length(ell);
N1 = size(x1,1);
N2 = size(x2,1);

Kpq = zeros(Q*N1, Q*N2);

for t1 = 1:N1
    for t2 = 1:N2
        rows = (t1-1)*Q + (1:Q);
        cols = (t2-1)*Q + (1:Q);
        for p = 1:Q
            for q = 1:Q
                Kpq(rows(p),cols(q)) = sqrt(2*ell(p).*ell(q)./(ell(p).^2+ell(q).^2)) * ...
                    exp(-norm(x1(t1,:)-x2(t2,:))^2 ./ (ell(p)^2+ell(q)^2));
            end
        end
    end
end
