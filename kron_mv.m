function b = kron_mv(As, x)

D = length(As);
G = zeros(D,1);
for d = 1:D
    Ad = As{d};
    assert(size(Ad,1) == size(Ad,2));
    G(d) = size(Ad,1);
end
N = prod(G);

b = x;
for dd = D:-1:1
    Ad = As{dd};
    X = reshape(b, G(dd), round(N/G(dd)));
    b = X'*Ad';
%     Y = Ad*X;
%     b = Y';
    b = b(:);
end