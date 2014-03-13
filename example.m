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
Omega = inv(Sigma);
[m, logLik] = minFunc(@simpleVariational, m0, optMinFunc, y, X, mu, Omega, v, bound);

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

