
SGD-2.0
-------

L. Bottou, October 2011



1. INTRODUCTION

The goal of this package is to illustrate the efficiency of stochastic
gradient descent for large-scale learning tasks.  

Two algorithms,

    * Stochastic gradient descent (SGD), and
    * Averaged stochastic gradient descent (ASGD),

are applied to two well known problems

    * Linear Support Vector Machines, and
    * Conditional Random Fields.

The corresponding programs are designed for simplicity and readability.  In
particular they avoid optimizations that would made the programs less
readable.  The sole exception is the handling of sparse training data.



2. DATASETS

The programs are demonstrated using a number of standard datasets.

* The RCV1-V2 dataset.
* The ALPHA and WEBSPAM datasets from the first Pascal Large Scale Learning Challenge.
* The dataset associated with the CONLL2000 chunking task.

These datasets are available from the web.  File "data/README" contains
instructions for downloading.  The Pascal datasets must be 
preprocessed using a relatively slow python script.



3. ALGORITHMS

Unlike most optimization algorithm, each iteration of these stochastic
algorithms process a single example and update the parameters.  Although the
theory calls for picking a random example at each iteration, this
implementation performs sequential passes over randomly shuffled training
examples. This process is in fact more effective in practice.  Each pass is
called an epoch.

Assume we have an objective function of the form
 
     Obj(w) = 1/2 lambda w^2  + 1/n sum_i=1^n L(z_i,w)

where w is the parameter, {z_1,...,z_n} are the training examples, 1/2 \lambda
w^2 a regularization term, and L(z,w) is the loss function. Each iteration of
the SGD algorithm picks a single example z and updates the parameter vector
using the formula:

    SGD:    w := (1 - lambda eta_t) w - eta_t dL/dw(z,w)

The trick of course is to choose the gain sequence eta_t wisely.  We use the
formula eta_t = eta_0 / (1 + lambda eta_0 t), and we pick eta_0 by trying
several gain values on a subset of the training data.  In order to leverage
sparse dataset, we represent vector w as the ratio of a vector W and a scalar
wDivisor, that is, w = W / wDivisor. Each iteration effectively becomes:

    SGD:    wDivisor = wDivisor / (1 - lambda eta_t)
            W = W - eta_t wDivisor dL/dw(z,w)

The ASGD algorithm maintains two parameter vectors. The first parameter
vector, w, is updated like the SGD parameter. However, the output of the
algorithm is the second parameter vector, a, which computes an average of 
all the previous values of w.

    ASGD:   w := (1 - lambda eta_t) w - eta_t dL/dw(z,w)
            a := a + mu_t [ w - a ]

This algorithm has been shown to work extremely well (Polyak and Juditsky,
1992) provided that the sequence eta_t decreases with exactly the right speed.
We follow (Xu, 2010) and choose eta_t = eta_0 / (1 + lambda eta0 t) ^ 0.75.
We select eta_0 by trying several gain values on a subset of the training
data, and we start the averaging process after a certain time, that is, 
mu_t = 1/max(1,t-t0). Following (Xu, 2010), sparse training data is treated
using the substitutions w = W / wDivisor and a = (A + wFraction W) / aDivisor.
The algorithm effectively becomes:

    ASGD:   wDivisor = wDivisor / (1 - eta_t * lambda)
            W = W - eta_t wDivisor dL/dw(z,w)
            A = A + eta_t wFraction wDivisor dL/dw(z,w)
            aDivisor = aDivisor / (1 - mu_t)
            wFraction = wFraction + mu_t aDivisor / wDivisor



4. SUPPORT VECTOR MACHINES

The directory "svm" contains programs to train a L2-regularized linear model
for binary classification tasks. Compilation time switches determine whether
the models include a bias term, whether the bias term is regularized, and
which loss function should be used.  The default is to use an unregularized
bias term using the log-loss function L(x,y,w) = log(1+exp(-ywx)). See file
"svm/README" for details about these programs and their usage for each of the
datasets.



5. CONDITIONAL RANDOM FIELDS

The directory "crf" contains programs "crfsgd" and "crfasgd" for training
conditional random fields for sequences. Both programs take data files and
template files and produces tagging files similar to those of Taku Kudo's
CRF++ program described at <http://crfpp.sourceforge.net/>.  However they also
accepts gzipped data files instead of plain files. See the file "crf/README"
for detailed information about these programs and their usage.





