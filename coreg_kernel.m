function [K,dK] = coreg(X1,X2,hyp)
%% Gives a kernel of form K = \sum_i z_i * z_i^T
% Dummy inputs X1, X2 to have consistent inputs with gausskernel.m
K = 1e-3 * eye(length(hyp.z{1}));
oz = ones(size(hyp.z{1}));
dK.z = cell(length(hyp.z), length(hyp.z{1}));
for r = 1:length(hyp.z)
    K = K + hyp.z{r} * hyp.z{r}';
    for d = 1:length(hyp.z{r})
        oz = 0*oz; oz(d) = 1;
        dK.z{r,d} = oz * hyp.z{r}' + hyp.z{r} * oz';
    end
end

