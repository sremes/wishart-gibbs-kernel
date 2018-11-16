% a very simple test case, N=100 time points, M=4 dimensions, S=20 observations, 
T = [(1:100)', (1:100)']; % input dimension D = 2
x = randn(100, 4, 20);
y = [ones(10, 1); zeros(10, 1)]; % S=10+10=20
ell_u = ones(3, 1); % Q=3 latent functions
ell_b = 1.0;
ell_z = 1.0;

model = lcgp(T, x, y, ell_u, ell_b, ell_z);
