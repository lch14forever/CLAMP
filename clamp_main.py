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

def parse_lsh_parms(parms):
    parm_list = parms.split(',')
    parm_parsed = {}
    for i in parm_list:
        p = i.split(':')
        parm_parsed[p[0]] = p[1]
    return parm_parsed


def main(arguments):
    elapsed_time_s = time()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-d", "--database",
                        required="True",
                        nargs="+",
                        metavar="database",
                        help="The database of training data")
    parser.add_argument("-q", "--query",
                        required="True",
                        dest="query",
                        help="The query dataset")
    parser.add_argument("-k", "--num_nearest_neighbors",
                        dest="k",
                        default = '-1',
                        type = int,
                        help="The number of nearest neighbors: [20 percent of the trainning, bounded by [10, 100]]")
    parser.add_argument("-l", "--lsh_method",
                        dest="lsh_method",
                        default = 'psd',
                        help="The lsh method used: psd (Euclidean)/rhp (Cosine) [psd]")
    parser.add_argument("-p", "--lsh_parms",
                        dest="lsh_parms",
                        default = 'M:521,L:15,T:2,W:5',
                        help="Parameters used by LSHBOX (parm1:val1,parm2:val2) [M:521,L:20,T:2,W:5]")
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
    
    for d in args.database:
        with open(d, 'rU') as f:
            for l in f:
                line = l.strip().split('\t')
                train_class_label.append(int(line[0]))
                train_dat.append(map(float,line[1:]))
    if args.k == -1:
        k = min( int(0.2*len(train_class_label)), 100 )
        k = max(k, 10)
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
    program_time = datetime.now().strftime('%m-%d_%H-%M-%S.%f')
    tmp_folder += program_time +  '/'
    
    os.makedirs(tmp_folder)

    sys.stderr.write(' '.join(['Using', tmp_folder , 'as temp folder ... \n']))
    sys.stderr.write('Number of Nearest Neighbours (K): %d \n' %(k))
    sys.stderr.write('LSH method: %s \n' %(args.lsh_method))
    sys.stderr.write('LSH parameters: %s \n' %(args.lsh_parms))
    sys.stderr.write('SVM parameters: %s \n' %(args.svm_parms))
    tmp_index = tmp_folder + 'lsh.index'
    
    index_time_s = time()
    lsh_parms = parse_lsh_parms(args.lsh_parms)
    if args.lsh_method == 'rhp':
        mat = libpylshbox.rhplsh()
        mat.init_mat(train_dat, tmp_index, int(lsh_parms['M']), int(lsh_parms['L']), int(lsh_parms['N']))
    elif args.lsh_method == 'psd':
        mat = libpylshbox.psdlsh()
        mat.init_mat(train_dat, tmp_index, int(lsh_parms['M']), int(lsh_parms['L']), int(lsh_parms['T']), float(lsh_parms['W']))
    else:
        os.rmdir(tmp_folder)
        sys.exit('Wrong LSH method! Use rhp or psd\n')


    index_time_e = time()
    accuracy  = 0
    with open(args.query, 'rU') as f:
        counter = 0
        T_counter = 0
        TP_counter = 0
        FP_counter = 0
        for l in f:
            counter += 1
            line = l.strip().split('\t')
            if args.isTesting:
                test_class_label = int(line[0])
                test_dat = map(float, line[1:])
                if test_class_label == 1:
                    T_counter += 1
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
                if test_class_label == 1 and p_label[0] == 1:
                    ## TP
                    TP_counter += 1
                    sys.stderr.write('Correct!\n')
                elif test_class_label == 0 and p_label[0] == 1:
                    ## FP
                    FP_counter += 1
                    sys.stderr.write('Wrong!\n')
                elif test_class_label == 1 and p_label[0] == 0:
                    ## FN
                    sys.stderr.write('Wrong!\n')
                else:
                    ## TN
                    sys.stderr.write('Correct!\n')
                    
            else:
                args.outfile.write("%d\n" %(p_label[0]))

        sys.stderr.write('=============================================================\n')

        if args.isTesting:
            sys.stderr.write("Number of test cases: %d \n" %(counter))
            recall = TP_counter*1.0/T_counter
            precision = TP_counter*1.0/(TP_counter+FP_counter)
            f1 = 2*recall*precision/(precision+recall)
            sys.stderr.write("Recall: %.2f \n" %(recall)) 
            sys.stderr.write("Precision: %.2f \n" %(precision)) 
            sys.stderr.write("F1: %.2f \n" %(f1)) 

    elapsed_time_e = time()
    elapsed_time = elapsed_time_e - elapsed_time_s
    index_time = index_time_e - index_time_s
    if args.isTesting:
        args.outfile.write("Recall: %.2f \n" %(recall))      
        args.outfile.write("Precision: %.2f \n" %(precision))
        args.outfile.write("F1: %.2f \n" %(f1))              
        args.outfile.write('Elapsed_Time\t%s\nw' %(elapsed_time))
        args.outfile.write('IndexTime\t%s\n' %(index_time))
        args.outfile.write('TrainingTime\t%s\n' %(total_train_time))
        args.outfile.write('TestingTime\t%s\n' %(total_test_time))    
    
    os.remove(tmp_index)
    os.rmdir(tmp_folder)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
