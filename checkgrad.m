function d = checkgrad(f, X, e, varargin);

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e, ...)
%
% where X is the argument and e is the small perturbation used for the finite
% differences. and the P1, P2, ... are optional additional parameters which
% get passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, ...)
%
% where fX is the function value and dfX is a vector of partial derivatives.
%
% Carl Edward Rasmussen, 2001-08-01.

[y dy] = feval(f, X, varargin{:});          % get the partial derivatives dy

Z = X; X = unwrap(X); dy = unwrap(dy);
dh = zeros(length(X),1) ;
for j = 1:length(X)
  dx = zeros(length(X),1);
  dx(j) = dx(j) + e;                            % perturb a single dimension
  y2 = feval(f, rewrap(Z, X+dx), varargin{:});
  y1 = feval(f, rewrap(Z, X-dx), varargin{:});
  dh(j) = (y2 - y1)/(2*e);
end

disp([dy dh])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum



function v = unwrap(s)
% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    v = [v; unwrap(s{i})];
  end
end                                                   % other types are ignored

function [s v] = rewrap(s, v)
% Map the numerical elements in the vector "v" onto the variables "s" which can
% be of any type. The number of numerical elements must match; on exit "v"
% should be empty. Non-numerical entries are just copied. See also unwrap.m.
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially 
    [s{i} v] = rewrap(s{i}, v);
  end
end                                             % other types are not processed
