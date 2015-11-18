CLAMP: Classification of time series using LAMP
=============
Authors:
--------------
Li Chenhao, Wang Xiyan, Xu Qian

Description:
--------------
CLAMP is a semi-lazy learning frame work designed for time series classification. Given a test data point, it firstly performs k-Nearest Neighbour (lazy learning), to find a subset of training data. The subset is then used for training a Support Vector Machine (eager learning) for the final classification.

CLAMP uses Local Sensitive Hashing based on Euclidean distance to perform fast kNN search.

CLAMP outperforms lazy learning and eager learning on more than 2/3 of the data in UCR time series classification archive.

Requirements:
--------------
 - Linux (Tested on Ubuntu 14.04), Windows (Tested on Windows 7 64-bits)
 - Python 2.7

Usage:
--------------
```
usage: clamp_main.py [-h] -d DATABASE -q QUERY [-k K] [-l LSH_METHOD]
                     [-f FEATURE] [-p LSH_PARMS] [-s SVM_PARMS] [--prediction]
                     [-o OUTFILE]

Classification of time-series using LAMP (CLAMP)

optional arguments:
  -h, --help            show this help message and exit
  -d DATABASE, --database DATABASE
                        The database of training time series
  -q QUERY, --query QUERY
                        The query time series
  -k K, --num_nearest_neighbors K
                        The number of nearest neighbors: [20 percent of the
                        trainning, bounded by [10, 100]]
  -l LSH_METHOD, --lsh_method LSH_METHOD
                        The lsh method used: psd (Euclidean)/rhp (Cosine)
                        [psd]
  -f FEATURE, --feature FEATURE
                        Feature based method: dtw (DTW distances), dwt
                        (Discrete Wavelet Transformation) [Raw data]
  -p LSH_PARMS, --lsh_parms LSH_PARMS
                        Parameters used by LSHBOX (parm1:val1,parm2:val2)
                        [M:521,L:20,T:2,W:5]
  -s SVM_PARMS, --svm_parms SVM_PARMS
                        Parameters used by LIBSVM (parm1:val1,parm2:val2)
                        [t:0]
  --prediction          If not specified, will treat the first column in query
                        as class label.
  -o OUTFILE, --outfile OUTFILE
                        Output file

```
Examples:

* Basic usage (testing mode):
```
 $ python clamp_main.py -d data/Gun_Point_TRAIN -q data/Gun_Point_TEST -o data/Gun_Point_results
```
* Using DTW distances as features:
```
 $ python clamp_main.py -d data/Gun_Point_TRAIN -q data/Gun_Point_TEST -o data/Gun_Point_results -f dtw -s t:2,q
```
* Run lazy learning with Euclidean distance
```
 $ python clamp_main.py -d data/Gun_Point_TRAIN -q data/Gun_Point_TEST -o data/Gun_Point_results -k 1
```
* Run prediction mode
```
 $ python clamp_main.py -d data/Gun_Point_TRAIN -q data/Gun_Point_TEST -o data/Gun_Point_predicted --prediction
```
