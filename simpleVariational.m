function [f,g] = simpleVariational(m, y, X, mu, Omega, v, bound)
% objective function for variational approx N(m, V) with a restriction that diagonal of V is 1.
% f: -0.5* (m-mu)'*Omega*(m-mu) + sum_i f_i(yi,mi,1)
% g: -Omega(m-mu)
%
% Written by Emtiyaz, EPFL
% Modified on March 8, 2014
  
  [fi, gmi, gvi] = ElogLik('bernLogit', y, m, v, bound);
  e = m-mu;
  g = Omega*e;
  f = -e'*g/2 + sum(fi);
  g = -g + gmi; 

  f = -f;
  g = -g;
