VariationalApproxExample
========================

An example of variational approximation for Gaussian process classification.
To run this code, download the code in a directory, and do following in MATLAB,
$ addpath(genpath(pwd))
$ example

We generate synthetic classification data (yi,Xi). We assume a GP prior with zero mean and a linear Kernel with a logit likelihood to generate binary data. 

As a simple example of variational approximation, we will fit an approximate Gaussian posterior N(m,V) with a restriction that the diagonal of V is 1. Our goal is to find m and V.

We will use the KL method of Kuss and Rasmussen, 2005.

We write the objective function as 

max_m f(m) = -0.5* (m-mu)'*Omega*(m-mu) + sum_i f_i(yi,mi,1)

where f_i = E(log p(yi|xi)) wrt N(xi|mi,1) of the logit likelihood p(yi|xi) = exp(yi*xi)/(1+exp(xi))

We use LBFGS implemented in minFunc by Mark Schmidt
http://www.di.ens.fr/~mschmidt/Software/minFunc.html

