#!/usr/bin/env python

"""Classification of time-series using LAMP (CLAMP)

"""

import os
import sys
import argparse
import libpylshbox
import fastdtw
from datetime import datetime
from time import time
from svmpy import svmutil as svm

def parse_svm_parms(parms):
    parm_list = parms.split(',')
    parm_parsed = []
    for i in parm_list:
        p = i.split(':')
        p[0] = '-'+p[0]
        parm_parsed.extend(p)
    return ' '.join(parm_parsed)

def dtw_dist_mat(dat):
    length = len(dat)
    dtw_distances = [[0 for x in xrange(length)] for x in xrange(length)]
    sys.stderr.write('Precomputing DTW distance matrix [%dx%d] for training set...\n' %(length, length))
    for i in xrange(length):
        sys.stderr.write('\r')
        sys.stderr.write("[%-50s] %d%%" % ('='*(50*(i+1)/length), (i+1)*100*1.0/length))
        sys.stderr.flush()
        sys.stderr.write("\n")
        for j in xrange(i+1,length):
            dist = (fastdtw.fastdtw(dat[i], dat[j]))[0]
            dtw_distances[i][j] = dtw_distances[j][i] = dist
    

def main(arguments):
    elapsed_time_s = time()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-d", "--database",
                        required="True",
                        dest="database",
                        help="The database of training time series")
    parser.add_argument("-q", "--query",
                        required="True",
                        dest="query",
                        help="The query time series")
    parser.add_argument("-k", "--num_nearest_neighbors",
                        dest="k",
                        default = '-1',
                        type = int,
                        help="The number of nearest neighbors: [100 or 20 percent of the trainning, whichever is smaller]")
    parser.add_argument("-l", "--lsh_method",
                        dest="lsh_method",
                        default = 'psd',
                        help="The lsh method used: psd (Euclidean)/rhp (Cosine) [psd]")
    parser.add_argument("-f", "--feature",
                        dest="feature",
                        default = None,
                        help="Feature based method: dtw (DTW distances) [None]")
    parser.add_argument("-s", "--svm_parms",
                        dest="svm_parms",
                        default = 't:0,q',
                        help="Parameters used by LIBSVM (parm1:val1,parm2:val2) [t:0]")
    parser.add_argument("--prediction",
                        action = "store_false",
                        dest="isTesting",
                        help="If not specified, will treat the first column in query as class label.")
    parser.add_argument('-o', '--outfile', help="Output file", dest = 'outfile',
                        default=sys.stdout, type=argparse.FileType('w'))
    
    ## read input file (convert it to a list)
    args = parser.parse_args(arguments)
    train_class_label = []
    train_dat = []

    with open(args.database, 'rU') as f:
        for l in f:
            line = l.strip().split(',')
            train_class_label.append(int(line[0]))
            train_dat.append(map(float,line[1:]))
    if args.k == -1:
        k = min( int(0.2*len(train_class_label)), 100 )
    else:
        k = args.k

    
    total_train_time = 0
    total_test_time = 0
    
    ## testing for temporary folder
    tmp_folder = './lshbox_tmp'
    if os.path.exists("/dev/shm"):
        tmp_folder = '/dev/shm/lshbox_tmp'
    elif os.path.exists("/tmp"):
        tmp_folder = '/tmp/lshbox_tmp'
    program_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    tmp_folder += program_time + '/'
    
    os.makedirs(tmp_folder)

    sys.stderr.write(' '.join(['Using', tmp_folder , 'as temp folder ... \n']))
    sys.stderr.write('Number of Nearest Neighbours (K): %d \n' %(k))
    sys.stderr.write('LSH method: %s \n' %(args.lsh_method))
    sys.stderr.write('SVM parameters: %s \n' %(args.svm_parms))
    sys.stderr.write('Features used: %s \n' %(args.feature))
    tmp_index = tmp_folder + 'lsh.index'

    index_time_s = time()
    if args.lsh_method == 'rhp':
        mat = libpylshbox.rhplsh()
        mat.init_mat(train_dat, tmp_index, 521, 5, 6)
    elif args.lsh_method == 'psd':
        mat = libpylshbox.psdlsh()
        mat.init_mat(train_dat, tmp_index, 521, 200, 2, 5)
    else:
        os.rmdir(tmp_folder)
        sys.exit('Wrong LSH method! Use rhp or psd\n')

    if args.feature == 'dtw': 
        ## use dtw distances as features
        ## precompute the distances
        dtw_distances = dtw_dist_mat(train_dat)

    index_time_e = time()
    accuracy  = 0
    with open(args.query, 'rU') as f:
        counter = 0
        correct_prediction = 0
        for l in f:
            counter += 1
            line = l.strip().split(',')
            if args.isTesting:
                test_class_label = int(line[0])
                test_dat = map(float, line[1:])
            else:
                test_dat = map(float, line)
                test_class_label = 1
                
            result = mat.query(test_dat, 2, k)
            kNN_index = result[0]

            kNN_labels = [ train_class_label[i] for i in kNN_index]

            sys.stderr.write("Processing test case %d\n" %(counter))

            ## if kNN are all of one class, just report that
            tmp_label = kNN_labels[0]
            if all(label == tmp_label for label in kNN_labels):
                p_label = [tmp_label]
                sys.stderr.write("All nearest neighbour belong to the same class %d, assign class lazily...\n" %(tmp_label))
            ## else run the eager learning part
            else:
                train_time_s = time()
                if args.feature == 'dtw':
                    ## retrieve dtw feature matrix
                    kNN_dat = []
                    for i in kNN_index:
                        kNN_dat.append([train_dat[i][x] for x in kNN_index])
                else:
                    kNN_dat = [ train_dat[i] for i in kNN_index ]
                
                ## training
                model = svm.svm_train(kNN_labels, kNN_dat, parse_svm_parms(args.svm_parms))
                train_time_e = time()
                total_train_time += train_time_e - train_time_s
                ## testing
                test_time_s = time()
                p_label, p_acc, p_val = svm.svm_predict([test_class_label], [test_dat], model ,'-q')
                test_time_e = time()
                total_test_time += test_time_e - test_time_s
            if args.isTesting:
                if test_class_label == p_label[0]:
                    correct_prediction += 1
                    sys.stderr.write('Correct!\n')
                else:
                    sys.stderr.write('Wrong!\n')
            else:
                args.outfile.write("%d\n" %(p_label[0]))

        sys.stderr.write('=============================================================\n')

        if args.isTesting:
            sys.stderr.write("Number of test cases: %d \n" %(counter))
            sys.stderr.write("Number of correct predictions: %d \n" %(correct_prediction))
            accuracy = correct_prediction*1.0/counter
            sys.stderr.write("Accuracy: %.2f \n" %(accuracy)) 

    elapsed_time_e = time()
    elapsed_time = elapsed_time_e - elapsed_time_s
    index_time = index_time_e - index_time_s
    if args.isTesting:
        args.outfile.write('Accuracy\t%f\n' %(accuracy))
        args.outfile.write('Elapsed_Time\t%s\n' %(elapsed_time))
        args.outfile.write('IndexTime\t%s\n' %(index_time))
        args.outfile.write('TrainingTime\t%s\n' %(total_train_time))
        args.outfile.write('TestingTime\t%s\n' %(total_test_time))    
    
    os.remove(tmp_index)
    os.rmdir(tmp_folder)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
