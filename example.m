% Demo for a simple Gaussian process approximation
% We generate synthetic classification data (yi,Xi)
% We will assume a GP prior with zero mean and a linear Kernel
% with a logit likelihood to generate binary data.
% We will fit an approximate Gaussian posterior N(m,V) 
% with a restriction that diagonal of V is 1.
% Our goal is to find such m and V.
% This corresponds to the KL method of Kuss and Rasmussen, 2005
% We can write the objective function as 
% max_m f(m) = -0.5* (m-mu)'*Omega*(m-mu) + sum_i f_i(yi,mi,1)
%     where f_i = E(log p(yi|xi)) wrt N(xi|mi,1) of logit likelihood
% We use LBFGS implemented in minFunc by Mark Schmidt
%
% Written by Emtiyaz, EPFL
% Modified on March 8, 2014

clear all;
close all;

% synthetic data
setSeed(1);
N = 100; % number of data examples
D = 10; % feature dimensionality
X = [5*rand(N/2,D); -5*rand(N/2,D)]; 
Sigma = X*X' + eye(N); % linear kernel
mu = zeros(N,1); % zero mean
y = mvnrnd(mu, Sigma, 1);
y = y(:) + 0*0.1*randn(N,1); % add some noise
y = (y(:)>0);

% optimizers options
optMinFunc = struct('Display', 1,...
    'Method', 'lbfgs',...
    'DerivativeCheck', 'off',...
    'LS', 2,...
    'MaxIter', 1000,...
    'MaxFunEvals', 1000,...
    'TolFun', 1e-4,......
    'TolX', 1e-4);

% load bound
load('llp.mat'); 

% optimize wrt m (see function simpleVariational.m for details)
m0 = mu; % initial value
v = ones(N,1); % fix v to 1
[m, logLik] = minFunc(@simpleVariational, m0, optMinFunc, y, X, mu, inv(Sigma), v, bound);

% plot
figure(1)
imagesc(Sigma); colorbar;
title('GP Kernel matrix');

figure(2)
stem(y);
hold on
plot(1./(1+exp(-m)), '*r','markersize', 10);
ylim([-0.05 1.05]);
ylabel('Prediction for training data');



