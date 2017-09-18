function [f_test,y_test,u_test] = lcm_predict(model, x_test)

x_test = reshape(permute(x_test, [2 1 3]), [model.M*model.N, size(x_test,3)]);
u_test = model.u_sigma * (model.tau*model.B'*x_test);
f_test = model.w' * u_test + model.b;

y_test = f_test > 0;