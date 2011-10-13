



1. COMPILING SVMSGD AND SVMASGD

Compiling under Unix is achieved using the traditional command "make".  
The compilation requires the libz library. This library usually comes
preinstalled on most Linux distributions, and is otherwise available
from from http://www.zlib.org.  

Compilation switches can be conveniently specified as follows:

$ make clean; make OPT='-DLOSS=HingeLoss'

The available loss functions can be looked up in file "loss.h".
The following compilation switches can be used:

    -DLOSS=<lossfunctionclass>    Select the loss function (LogLoss).
    -DBIAS=<0_or_1>               Select whether the model has bias (yes).
    -DREGULARIZED_BIAS=<0_or_1>   Select whethet the bias is regularized (no).

Compiling under Windows is possible using Cygwin, using MSYS, or using the
MSVC project files provided in the subdirectory "win" of the sgd distribution.
Make sure to read the instructions as you need to compile zlib adequately.
You then need to copy the executable files in this directory.



2. USAGE

Synopsis:

    svmsgd [options] trainfile [testfile]
    svmasgd [options] trainfile [testfile]

Programs "svmsgd" and "svmasgd" compute a L2 regularized linear model using
respectively the SGD and ASGD algorithms. Both programs perform a number of
predefined training epochs over the training set. Each epoch is followed by a
performance evaluation pass over the training set and optionally over a
validation set. The training set performance is useful to monitor the progress
of the optimization. The validation set performance is useful to estimate the
generalization performance. The recommended procedure is to monitor the
validation performance and stop the algorithm when the validation metrics no
longer improve. In the limit of large number of examples, program "svmasgd"
should reach this point after one or two epochs only.

Both programs accept the same options:

    -lambda x       : Regularization parameter (default: 1e-05.)
    -epochs n       : Number of training epochs (default: 5)
    -dontnormalize  : Do not normalize the L2 norm of patterns.
    -maxtrain n     : Restrict training set to the first n examples.

Program svmasgd accepts one additional option:

    -avgstart x       : Starts averaging after n iterations (default: 1.0.)

The programs assume that the training data file already contains randomly
shuffled examples. In addition, unless option -dontnormalize is specified,
every input vector is scaled to unit norm when it is loaded.

Several kinds of data files are supported:

  * Text files in svmlight format (suffix ".txt"),
  * Dedicated binary files (suffix ".bin"),
  * Gzipped versions of the above (suffix ".txt.gz" or ".bin.gz").



3. PREPROCESSING

Please follow the instructions in file "data/README.txt" to populate the
directories "data/rcv1" and "data/pascal". Three preprocessing programs can
then be used to generate training and validation data files.
These programs take no argument. 

 * Program "prep_alpha" preprocesses the alpha dataset: loading the original
   training data file, applying a random permutation, and producing suitable
   training and validation files named "alpha.train.bin.gz" and
   "alpha.test.bin.gz".

 * Program "prep_webspam" preprocesses the webspam dataset: loading the
   original training data file, applying a random permutation, and producing
   suitable training and validation files named "webspam.train.bin.gz" and
   "webspam.test.bin.gz".  You probably need 16GB of RAM for this dataset.

 * Program "prep_rcv1" preprocesses the RCV1-V2 dataset. The task consists of
   identifying documents belonging to the CCAT category. In order to obtain a
   larger training set, the official training and testing set are swapped: the
   four official test files become the training set, and the official training
   file becomes the validation set.  Program "prep_rcv1" therefore recomputes
   the TF-IDF features in order to base the IDF coefficients on our new
   training set. As usual the data is randomly shuffled before producing the
   data files "rcv1.train.bin.gz" and "rcv1.test.bin.gz".



4. EXAMPLE: RCV1-V2, HINGE LOSS

Preparation

    $ ./prep_rcv1
    $ make clean && make OPT=-DLOSS=HingeLoss


Using stochastic gradient descent (svmsgd):

    $ ./svmsgd -lambda 1e-4 rcv1.train.bin.gz rcv1.test.bin.gz

    # Running: ./svmsgd -lambda 0.0001 -epochs 5
    # Compiled with:  -DLOSS=HingeLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=0.5
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=1141.36 wBias=0.0690558
    Total training time 0.2 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170795958407 Cost=0.227863744764 Misclassification=5.686%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187133783125 Cost=0.244201569483 Misclassification=6.061%.
    --------- Epoch 2.
    Training on [0, 781264].
    wNorm=1141 wBias=0.0709104
    Total training time 0.39 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170593793946 Cost=0.227643679517 Misclassification=5.671%.
    test:  Testing on [0, 23148].
    test:  Loss=0.18709754618 Cost=0.244147431751 Misclassification=6.043%.
    --------- Epoch 3.
    Training on [0, 781264].
    wNorm=1140.74 wBias=0.0715381
    Total training time 0.58 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17054568518 Cost=0.227582476075 Misclassification=5.668%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187105426667 Cost=0.244142217562 Misclassification=6.035%.
    --------- Epoch 4.
    Training on [0, 781264].
    wNorm=1140.62 wBias=0.0715058
    Total training time 0.77 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17051783095 Cost=0.227548613852 Misclassification=5.67%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187110187534 Cost=0.244140970436 Misclassification=6.039%.
    --------- Epoch 5.
    Training on [0, 781264].
    wNorm=1140.58 wBias=0.0715744
    Total training time 0.97 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170506173396 Cost=0.227534977776 Misclassification=5.67%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187108951994 Cost=0.244137756374 Misclassification=6.026%.


Using averaged stochastic gradient descent (svmasgd):

    $ ./svmasgd -lambda 1e-4 - epochs 3 rcv1.train.bin.gz rcv1.test.bin.gz

    # Running: ./svmasgd -lambda 0.0001 -epochs 3 -avgstart 1
    # Compiled with:  -DLOSS=HingeLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=0.5
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=1151.98 aNorm=1151.98 wBias=0.0660923 aBias=0.0660923
    Total training time 0.28 secs.
    train: Testing on [0, 781264].
    train: Loss=0.171361609048 Cost=0.22896057145 Misclassification=5.711%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187480624465 Cost=0.245079586867 Misclassification=6.065%.
    --------- Epoch 2.
    Training on [0, 781264].
    wNorm=1145.12 aNorm=1146.24 wBias=0.0685983 aBias=0.0710148
    Total training time 0.62 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17026402405 Cost=0.22751994446 Misclassification=5.672%.
    test:  Testing on [0, 23148].
    test:  Loss=0.186684175033 Cost=0.243940095443 Misclassification=6.022%.
    --------- Epoch 3.
    Training on [0, 781264].
    wNorm=1143.77 aNorm=1145.23 wBias=0.0701258 aBias=0.0708836
    Total training time 0.96 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17029659069 Cost=0.227485158805 Misclassification=5.673%.
    test:  Testing on [0, 23148].
    test:  Loss=0.186730744461 Cost=0.243919312577 Misclassification=6.026%.


The same experiment has been run using well known SvmLight and SvmPerf
software packages (Joachims, 1999, 2006). The experiments above use a
regularization coefficient lambda=1e-4.  Although this specific value copies
the settings described by Joachims (2006), the svm_light and svm_perf command
line arguments specify the regularization coefficient using different
calculations. The equivalent commands are:

    $ svm_light -c .0127998  train.dat svmlight.model
    training time: 23642 seconds
    test error: 6.0219%
    primal:0.227488

    $ svm_perf -c 100 train.dat svmperf.model
    training time: 66 seconds.
    test error: 6.0348%
    primal: 0.2278 

And using the LibLinear's dual coordinate ascent method (Hsieh, 2008)

    $ ./liblinear-1.8/train -B 1 -s 3 -c 0.0127998 rcv1.train.txt model
    training time: 2.50 seconds
    test error: 6.0219%




4. EXAMPLE: RCV1-V2, LOG LOSS

Preparation

    $ make clean && make OPT='-DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=1'


Using stochastic gradient descent (svmsgd):

    $ ./svmsgd -lambda 5e-7 -epochs 12 rcv1.train.bin.gz rcv1.test.bin.gz 

    # Running: ./svmsgd -lambda 5e-07 -epochs 12
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=16
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=63685.6 wBias=0.791762
    Total training time 0.35 secs.
    train: Testing on [0, 781264].
    train: Loss=0.1322143834 Cost=0.14813577228 Misclassification=4.87%.
    test:  Testing on [0, 23148].
    test:  Loss=0.157003248512 Cost=0.172924637392 Misclassification=5.659%.
    ...
    --------- Epoch 4.
    Training on [0, 781264].
    wNorm=48535.2 wBias=0.793385
    Total training time 1.36 secs.
    train: Testing on [0, 781264].
    train: Loss=0.11867338965 Cost=0.130807180982 Misclassification=4.329%.
    test:  Testing on [0, 23148].
    test:  Loss=0.143717091009 Cost=0.155850882341 Misclassification=5.27%.
    ...
    --------- Epoch 8.
    Training on [0, 781264].
    wNorm=46647 wBias=0.816894
    Total training time 2.71 secs.
    train: Testing on [0, 781264].
    train: Loss=0.117143686892 Cost=0.128805431703 Misclassification=4.255%.
    test:  Testing on [0, 23148].
    test:  Loss=0.142313428629 Cost=0.153975173439 Misclassification=5.179%.
    ...
    --------- Epoch 12.
    Training on [0, 781264].
    wNorm=46077.1 wBias=0.830094
    Total training time 4.06 secs.
    train: Testing on [0, 781264].
    train: Loss=0.116796617747 Cost=0.128315901686 Misclassification=4.243%.
    test:  Testing on [0, 23148].
    test:  Loss=0.141969857505 Cost=0.153489141444 Misclassification=5.141%.


Using averaged stochastic gradient descent (svmasgd):

    $ ./svmasgd -lambda 5e-7 -epochs 8 rcv1.train.bin.gz rcv1.test.bin.gz 

    # Running: ./svmasgd -lambda 5e-07 -epochs 8 -avgstart 1
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=16
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=80053.3 aNorm=80053.3 wBias=0.804545 aBias=0.804545
    Total training time 0.42 secs.
    train: Testing on [0, 781264].
    train: Loss=0.147851677288 Cost=0.167865001507 Misclassification=5.367%.
    test:  Testing on [0, 23148].
    test:  Loss=0.172513932416 Cost=0.192527256635 Misclassification=6.177%.
    ...
    --------- Epoch 4.
    Training on [0, 781264].
    wNorm=53203 aNorm=55096.5 wBias=0.763032 aBias=0.931136
    Total training time 2.19 secs.
    train: Testing on [0, 781264].
    train: Loss=0.115978351119 Cost=0.129279090376 Misclassification=4.222%.
    test:  Testing on [0, 23148].
    test:  Loss=0.141752793499 Cost=0.155053532757 Misclassification=5.158%.
    ...
    --------- Epoch 8.
    Training on [0, 781264].
    wNorm=49069.1 aNorm=51592.7 wBias=0.777501 aBias=0.902509
    Total training time 4.54 secs.
    train: Testing on [0, 781264].
    train: Loss=0.115864047594 Cost=0.128131330797 Misclassification=4.212%.
    test:  Testing on [0, 23148].
    test:  Loss=0.14134056649 Cost=0.153607849693 Misclassification=5.136%.


The same experiment has been run using the liblinear package.

 - Using the tron optimizer:

    $ ./liblinear-1.8/train -B 1 -s 0 -c 2.55994 rcv1.train.txt model
    training time: 33.40 seconds
    test error: 5.137%

 - Using the dual coordinate ascent optimizer:

    $ ./liblinear-1.8/train -B 1 -s 7 -c 2.55994 rcv1.train.txt model
    training time: 15.18 seconds
    test error: 5.128%


5. EXAMPLE: ALPHA, LOG LOSS

Preparation

    $ ./prep_alpha
    $ make clean && make

Using stochastic gradient descent:

    $ ./svmsgd -lambda 1e-6 -epochs 20 alpha.train.bin.gz alpha.test.bin.gz

    # Running: ./svmsgd -lambda 1e-06 -epochs 20
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file alpha.train.bin.gz
    # Read 124703+125297=250000 examples.
    # Reading file alpha.test.bin.gz
    # Read 124810+125190=250000 examples.
    # Number of features 501.
    # Using eta0=2
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=16230 wBias=0.0473942
    Total training time 0.52 secs.
    train: Testing on [0, 249999].
    train: Loss=0.548678268381 Cost=0.556793250617 Misclassification=26.82%.
    test:  Testing on [0, 249999].
    test:  Loss=0.552432284339 Cost=0.560547266575 Misclassification=27.01%.
    --------- Epoch 2.
    Training on [0, 249999].
    wNorm=15412.6 wBias=0.0113468
    Total training time 1.05 secs.
    train: Testing on [0, 249999].
    train: Loss=0.531136460583 Cost=0.538842772318 Misclassification=26.19%.
    test:  Testing on [0, 249999].
    test:  Loss=0.534786430567 Cost=0.542492742303 Misclassification=26.35%.
    ...
    --------- Epoch 5.
    Training on [0, 249999].
    wNorm=14117 wBias=-0.0346802
    Total training time 2.6 secs.
    train: Testing on [0, 249999].
    train: Loss=0.505480888329 Cost=0.512539404514 Misclassification=24.91%.
    test:  Testing on [0, 249999].
    test:  Loss=0.509024633921 Cost=0.516083150105 Misclassification=25.09%.
    ...
    --------- Epoch 20.
    Training on [0, 249999].
    wNorm=13065.9 wBias=-0.0685243
    Total training time 10.36 secs.
    train: Testing on [0, 249999].
    train: Loss=0.473929319282 Cost=0.480462269699 Misclassification=22.59%.
    test:  Testing on [0, 249999].
    test:  Loss=0.477352904632 Cost=0.48388585505 Misclassification=22.79%.


Using averaged stochastic gradient descent:

    $ ./svmasgd -lambda 1e-6 -epochs 5 alpha.train.bin.gz alpha.test.bin.gz

    # Running: ./svmasgd -lambda 1e-06 -epochs 5 -avgstart 1
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file alpha.train.bin.gz
    # Read 124703+125297=250000 examples.
    # Reading file alpha.test.bin.gz
    # Read 124810+125190=250000 examples.
    # Number of features 501.
    # Using eta0=2
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=16729.4 aNorm=16729.4 wBias=0.0618375 aBias=0.0618375
    Total training time 0.58 secs.
    train: Testing on [0, 249999].
    train: Loss=0.555545534991 Cost=0.563910246905 Misclassification=27%.
    test:  Testing on [0, 249999].
    test:  Loss=0.559338201533 Cost=0.567702913447 Misclassification=27.2%.
    --------- Epoch 2.
    Training on [0, 249999].
    wNorm=16024.8 aNorm=16177.5 wBias=0.031884 aBias=-0.0585204
    Total training time 1.45 secs.
    train: Testing on [0, 249999].
    train: Loss=0.464518389064 Cost=0.472530779201 Misclassification=21.65%.
    test:  Testing on [0, 249999].
    test:  Loss=0.468675093306 Cost=0.476687483443 Misclassification=21.87%.
    ...
    --------- Epoch 5.
    Training on [0, 249999].
    wNorm=14707.9 aNorm=15405.6 wBias=-0.0128081 aBias=-0.0563722
    Total training time 4.06 secs.
    train: Testing on [0, 249999].
    train: Loss=0.463992480808 Cost=0.471346417331 Misclassification=21.65%.
    test:  Testing on [0, 249999].
    test:  Loss=0.467999286155 Cost=0.475353222679 Misclassification=21.86%.




6. EXAMPLE: WEBSPAM, LOG LOSS

This dataset has a very high number of features.
We recommend having 16GB of ram for this.

Preparation. 

    $ ./prep_webspam
    $ make clean && make

Using stochastic gradient descent:

    $ ./svmsgd -lambda 1e-7 -epochs 10 webspam.train.bin.gz webspam.test.bin.gz

    # Running: ./svmsgd -lambda 1e-07 -epochs 10
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file webspam.train.bin.gz
    # Read 151592+98408=250000 examples.
    # Reading file webspam.test.bin.gz
    # Read 60597+39403=100000 examples.
    # Number of features 16609144.
    # Using eta0=16
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=291558 wBias=-3.02123
    Total training time 8.33 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0283830636612 Cost=0.0429609680624 Misclassification=0.8048%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0382940134873 Cost=0.0528719178885 Misclassification=0.987%.
    --------- Epoch 2.
    Training on [0, 249999].
    wNorm=282330 wBias=-2.27695
    Total training time 15.98 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0228423542354 Cost=0.0369588522914 Misclassification=0.6868%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0328592060835 Cost=0.0469757041394 Misclassification=0.914%.
    ...
    --------- Epoch 5.
    Training on [0, 249999].
    wNorm=225547 wBias=-1.47127
    Total training time 38.92 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0153457080155 Cost=0.0266230464572 Misclassification=0.3688%.
    test:  Testing on [0, 99999].
    test:  Loss=0.023982682269 Cost=0.0352600207107 Misclassification=0.623%.
    ...
    --------- Epoch 10.
    Training on [0, 249999].
    wNorm=194203 wBias=-1.24647
    Total training time 77.14 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0133282634555 Cost=0.023038408598 Misclassification=0.2752%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0212322092641 Cost=0.0309423544067 Misclassification=0.509%.


Using averaged stochastic gradient descent:

    $ ./svmasgd -lambda 1e-7 -epochs 10 webspam.train.bin.gz webspam.test.bin.gz

    # Running: ./svmasgd -lambda 1e-07 -epochs 10 -avgstart 1
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file webspam.train.bin.gz
    # Read 151592+98408=250000 examples.
    # Reading file webspam.test.bin.gz
    # Read 60597+39403=100000 examples.
    # Number of features 16609144.
    # Using eta0=16
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=314666 aNorm=314666 wBias=-3.12885 aBias=-3.12885
    Total training time 8.5 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0397637960172 Cost=0.0554970710172 Misclassification=1.25%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0507697914731 Cost=0.0665030664731 Misclassification=1.451%.
    --------- Epoch 2.
    Training on [0, 249999].
    wNorm=314240 aNorm=311720 wBias=-2.49689 aBias=-2.39171
    Total training time 21.47 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0170891660475 Cost=0.03280114886 Misclassification=0.5192%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0255372075546 Cost=0.0412491903671 Misclassification=0.672%.
    ...
    --------- Epoch 5.
    Training on [0, 249999].
    wNorm=250831 aNorm=282440 wBias=-1.59406 aBias=-1.91842
    Total training time 59.68 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0131488504686 Cost=0.0256903840623 Misclassification=0.4144%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0216984257223 Cost=0.0342399593161 Misclassification=0.596%.
    ...
    --------- Epoch 10.
    Training on [0, 249999].
    wNorm=210251 aNorm=247737 wBias=-1.32564 aBias=-1.60247
    Total training time 123.35 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0118189522858 Cost=0.0223314905671 Misclassification=0.358%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0200948107718 Cost=0.030607349053 Misclassification=0.551%.
