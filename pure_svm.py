#!/usr/bin/env python

"""Classification of time-series using LAMP (CLAMP)

"""

import os
import sys
import argparse
from datetime import datetime
from time import time
from svmpy import svmutil as svm

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def ec_query(test_dat, train_dat, k):
    dist = []
    for train in train_dat:
        dist.append(sum([ (x-y)**2 for x,y in zip(train, test_dat) ]))
    return argsort(dist)[0:k]

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
    
    total_train_time = 0
    total_test_time = 0
    


    accuracy  = 0
    with open(args.query, 'rU') as f:
        counter = 0
        correct_prediction = 0
        train_time_s = time()
        ## training
        model = svm.svm_train(train_class_label, train_dat, '-t 0 -q')
        train_time_e = time()
        total_train_time += train_time_e - train_time_s
        for l in f:
            counter += 1
            line = l.strip().split(',')
            if args.isTesting:
                test_class_label = int(line[0])
                test_dat = map(float, line[1:])
            else:
                test_dat = map(float, line)
                
                
            sys.stderr.write("Processing test case %d\n" %(counter))

            ## else run the eager learning part
                
            ## testing
            test_time_s = time()
            p_label, p_acc, p_val = svm.svm_predict([test_class_label], [test_dat], model)
            test_time_e = time()
            total_test_time += test_time_e - test_time_s
            
            if test_class_label == p_label[0]:
                correct_prediction += 1

        sys.stderr.write('=============================================================\n')
        sys.stderr.write("Number of test cases: %d \n" %(counter))    
        sys.stderr.write("Number of correct predictions: %d \n" %(correct_prediction))
        accuracy = correct_prediction*1.0/counter
        sys.stderr.write("Accuracy: %.2f \n" %(accuracy)) 

    elapsed_time_e = time()
    elapsed_time = elapsed_time_e - elapsed_time_s
    
    args.outfile.write('Accuracy\t%f\n' %(accuracy))
    args.outfile.write('Elapsed_Time\t%s\n' %(elapsed_time))
    args.outfile.write('TrainingTime\t%s\n' %(total_train_time))
    args.outfile.write('TestingTime\t%s\n' %(total_test_time))    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
