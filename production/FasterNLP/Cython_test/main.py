import test
import test_cython
import cProfile

print("------------- without cython : -------------")
cProfile.run('test.sumint_1_to_n(500000000)')

print("------------- with cython : -------------")
cProfile.run('test_cython.sumint_1_to_n(500000000)')

print("------------- with cython and cpdef : -------------")
cProfile.run('test_cython.sumint_1_to_n_cdef(500000000)')

