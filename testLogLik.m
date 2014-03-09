% Written by Emtiyaz, EPFL
% Modified on March 8, 2014
clear all
% poisson
m = 0.1*[-4:5]'; v = 0.1*[1:10]';
y = floor(5*rand(length(m),1));
[f, gm, gv] = ElogLik('poisson', y, m, v, []);

% Bernolli-logit
y = [rand(length(m),1)>0.5];
load('llp.mat'); % this should go into a look up table
[f, gm, gv] = ElogLik('bernLogit', y, m, v, bound);
