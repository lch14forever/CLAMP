## int DTWDistance(s: array [1..n], t: array [1..m], w: int) {
##     DTW := array [0..n, 0..m]
 
##    w := max(w, abs(n-m)) // adapt window size (*)
 
##     for i := 0 to n
##         for j:= 0 to m
##             DTW[i, j] := infinity
##     DTW[0, 0] := 0

##     for i := 1 to n
##         for j := max(1, i-w) to min(m, i+w)
##             cost := d(s[i], t[j])
##             DTW[i, j] := cost + minimum(DTW[i-1, j  ],    // insertion
##                                         DTW[i, j-1],    // deletion
##                                         DTW[i-1, j-1])    // match
 
##     return DTW[n, m]

# def minimum (a,b):
#     return min(a,b)
from libc.stdlib cimport malloc, free

def dtw(ts_a, ts_b):
    cdef int nrow = len(ts_a) + 1
    cdef int ncol = len(ts_b) + 1 
    #cdef double DTW[nrow][ncol]
    cdef double *DTW = <double *>malloc(nrow * ncol * sizeof(double))
    cdef double dist = 0.0
    for i in xrange(nrow):
        for j in xrange(ncol):
            DTW[i*nrow+j] = 3E300

    for i in xrange(1, nrow):
        for j in xrange(1, ncol):
            cost = (ts_a[i-1] - ts_b[j-1]) * (ts_a[i-1] - ts_b[j-1])
            DTW[ i*nrow+j ] = cost + min(DTW[(i-1)*nrow+j], DTW[i*nrow+j-1], DTW[(i-1)*nrow+j-1])
            # DTW[i][j] = cost + min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
    dist = DTW[(nrow-1) * nrow + ncol-1]
    free( DTW )
    return dist


# def dtw_dist_mat(twod_list):
#     cdef int nrow = len(twod_list) + 1
#     cdef double[:,:] DTW_dist = np.zeros((nrow, nrow))
#     for i in xrange(nrow - 1):
#         for j in xrange(i+1, nrow - 1 ):
#             dist = dtw(twod_list[i], twod_list[j])
#             DTW_dist[i][j] = DTW_dist[j][i] = dist
#     return list(DTW_dist)
