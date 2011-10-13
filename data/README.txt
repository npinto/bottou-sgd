
This directory should be 
populated with various data files
containing well known datasets.


* The following Reuters RCV1 dataset available from
  http://jmlr.csail.mit.edu/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

        rcv1/lyrl2004_tokens_test_pt0.dat.gz
        rcv1/lyrl2004_tokens_test_pt1.dat.gz
        rcv1/lyrl2004_tokens_test_pt2.dat.gz
        rcv1/lyrl2004_tokens_test_pt3.dat.gz
        rcv1/lyrl2004_tokens_train.dat.gz
        rcv1/rcv1-v2.topics.qrels.gz


* The following CONLL2000 data available from
  http://www.cnts.ua.ac.be/conll2000/chunking

        conll2000/train.txt.gz
        conll2000/test.txt.gz


* The following PASCAL data available from
  ftp://largescale.ml.tu-berlin.de/largescale/ 

        pascal/alpha_train.dat.bz2
        pascal/alpha_train.lab.bz2
        pascal/webspam_train.dat.bz2
        pascal/webspam_train.lab.bz2
        pascal/convert.py

   These files must then be decoded using the python script convert.py.
   This can take a while.
        $ cd pascal
        $ ./convert.py -o alpha.txt alpha train
        $ ./convert.py -o webspam.txt webspam train

