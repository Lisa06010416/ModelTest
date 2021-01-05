# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1

def sumint_1_to_n(I: int):
    s = 0
    for i in range(I):
        s += i
    return s


# cpdef => 可以被ㄘpython檔案呼叫
# cdef => 只能被cython檔案呼叫
cpdef int sumint_1_to_n_cdef(int I):
    cdef int s = 0
    cdef int i
    for i in range(I):
        s +=  i
    return s