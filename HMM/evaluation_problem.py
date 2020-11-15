# Evaluation Problem: Forward-backward Algorithm
# 看到一個觀察序列 o₁ o₂ ...... oT ，但是看不到狀態序列 s₁ s₂ ...... sT 的情況下，找出所有可能的路徑的機率的總和

# 有3種state s1 - s3, 3種output o1 - o3, 走3步t1-t3
# translation matric
#    s1 s2 s3
# s1 x1 x2 x3
# s2 y1 y2 y3
# s3 z1 z2 z3

# observe probability
#    o1 o2 o3
# s1 a  d  g
# s2 b  e  h
# s3 c  f  i


# 在只有3個statue且走3步的情況下，觀測到o1o2o3的機率 :
# [(adg+bdg+cdg)+(aeg+beg+ceg)+(afg+bfg+cfg)] + [(adh+bdh+cdh)+(aeh+beh+ceh)+(afh+bfh+cfh)] + [(adi+bdi+cdi)+(aei+bei+cei)+(afi+bfi+cfi)]
# g[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)] + h[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)] + i[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)]
# 可以看到後面[]的項都是可以重複利用的，且計算後面步數的機率需要前面的資訊 => dynamic programming


# dynamic programming :
#     o1    o2                      o3
# s1  a  d(a+b+c)   g[(d(a+b+c)+e(a+b+c)+f(a+b+c)]
# s2  b  e(a+b+c)   h[(d(a+b+c)+e(a+b+c)+f(a+b+c)]
# s3  c  f(a+b+c)   i[(d(a+b+c)+e(a+b+c)+f(a+b+c)]

# dynamic programming with translation weight:
#     o1    o2                      o3
# s1  a   l = d(a*x1+b*y1+c*z1)   o = g(l*x1+m*y1+n*z1)
# s2  b   m = e(a*x2+b*y2+c*z2)   p = h(l*x2+m*y2+n*z2)
# s3  c   n = f(a*x3+b*y3+c*z3)   q = i(l*x3+m*y3+n*z3)


import numpy as np

def eval_dp(S_index, O_index, observe_probability, transition_matrix, step):
    m = len(S_index)
    n = step
    dp = np.zeros((m, n))

    # init
    for i in range(m):
        dp[i][0] = observe_probability[S_index[i]][O_index[i]]
    # dp
    for j in range(1,n):
        for i in range(m):
            print("~~~~~")
            print(i)
            print(j)
            # print(dp)
            print(dp[:, j-1])
            print(transition_matrix[:, i])
            print(dp[:, j-1] * transition_matrix[:, i])
            prior = np.sum(dp[:, j-1] * transition_matrix[:, i])
            op = observe_probability[S_index[i]][O_index[j]]
            dp[i][j] = op * prior
    return dp



S_index = [0,1,2]
O_index = [1,0,2]
observe_probability = np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
transition_matrix = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
# transition_matrix = np.array([[1,1,1],[1,1,1],[1,1,1]])
step = 3

dp = eval_dp(S_index, O_index, observe_probability, transition_matrix, step)
print(dp)
