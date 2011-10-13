
This directory contains older variants of the basic svmsgd code.
These programs have not been updated to the sgd-2.0 standards
but should nevertheless work.

* svmcg compute the svm solution using primal 
  batch conjugate gradient method (Chapelle)

* svmolbfgs is an implementation of the online limited 
  storage BFGS (Shraudolph et al.)

* svmsgd2 is an alternative implementation of sgd for sparse
  dataset using different schedules for the updates associated
  with the loss term and the update associated with the
  regularization term.  

* svmsgdqn is a diagonal quasi-newton algorithm (Bordes et al.)
  with sometimes good but often inconsistent performance. 
  Using svmasgd is usually a better choice.
  
