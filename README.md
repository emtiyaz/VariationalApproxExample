VariationalApproxExample
========================

An example of variational approximation for Gaussian process classification.
To run this code, download the code in a directory, and do following in MATLAB,

$ addpath(genpath(pwd)) <br>
$ example

We generate synthetic classification data (yi,Xi). We assume a GP prior with zero mean and a linear Kernel with a logit likelihood to generate binary data. 

As a simple example of variational approximation, we will fit an approximate Gaussian posterior N(m,V) with a restriction that the diagonal of V is 1. Our goal is to find m and V.

We will use the KL method of <a href=http://eprints.pascal-network.org/archive/00001193/01/kuss05a.pdf>Kuss and Rasmussen, 2005</a> and solve the following optimization problem to find m: 

max_m f(m) = -(m-mu)'*Omega*(m-mu)/2 + sum_i fi(yi,mi,1)

where Omega is inverse of GP covariance matrix, mu is the mean, and fi = E(log p(yi|xi)) wrt N(xi|mi,1) of the logit likelihood p(yi|xi) = exp(yi*xi)/(1+exp(xi))

We use LBFGS implemented in minFunc by Mark Schmidt
http://www.di.ens.fr/~mschmidt/Software/minFunc.html

The implementation of fi for logit likelihood is based on the following
<a href=http://www.cs.ubc.ca/~emtiyaz/papers/paper-ICML2011.pdf>ICML paper</a> (also see the 
<a href=http://www.cs.ubc.ca/~emtiyaz/papers/truncatedGaussianMoments.pdf>Appendix</a>
)

