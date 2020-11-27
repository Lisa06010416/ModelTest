# learn from :
# http://web.ntnu.edu.tw/~algo/HiddenMarkovModel.html

# Evaluation Problem: Forward-backward Algorithm
# 看到一個觀察序列 o₁ o₂ ...... oT ，但是看不到狀態序列 s₁ s₂ ...... sT 的情況下，找出所有可能的路徑的機率的總和

# 有3種state s1 - s3, 3種output o1 - o3, 走3步t1-t3

# translation matric
#           s1   s2   s3
# init_s0   in1  in2  in3
# s1        x1   x2   x3
# s2        y1   y2   y3
# s3        z1   z2   z3

# observe probability
#    o1 o2 o3
# s1 a  d  g
# s2 b  e  h
# s3 c  f  i


# 在只有3個statue且走3步的情況下，觀測到o1o2o3的機率 :
# [(adg+bdg+cdg)+(aeg+beg+ceg)+(afg+bfg+cfg)] + [(adh+bdh+cdh)+(aeh+beh+ceh)+(afh+bfh+cfh)] + [(adi+bdi+cdi)+(aei+bei+cei)+(afi+bfi+cfi)]
# g[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)] + h[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)] + i[(ad+bd+cd)+(ae+be+ce)+(af+bf+cf)]
# 可以看到後面[]的項都是可以重複利用的，且計算後面步數的機率需要前面的資訊 => dynamic programming

# ----------- dynamic programming forward alpha: -----------
# alpha(s = 1,t =3) - 在第3個step，state s1，吐出o3的可能的機率 => g[(d(a+b+c)+e(a+b+c)+f(a+b+c)]
#     o1    o2                      o3
#     t1    t2                      t3
# s1  a  d(a+b+c)   g[(d(a+b+c)+e(a+b+c)+f(a+b+c)]
# s2  b  e(a+b+c)   h[(d(a+b+c)+e(a+b+c)+f(a+b+c)]
# s3  c  f(a+b+c)   i[(d(a+b+c)+e(a+b+c)+f(a+b+c)]

# dynamic programming with translation weight:
#         o1                    o2                               o3
# s1  a' = in1*a   d' = d(a'*x1 + b'*y1 + c'*z1)   g' = g(d'*x1 + e'*y1 + f'*z1)
# s2  b' = in2*b   e' = e(a'*x2 + b'*y2 + c'*z2)   h' = h(d'*x2 + e'*y2 + f'*z2)
# s3  c' = in3*c   f' = f('a*x3 + b'*y3 + c'*z3)   i' = i(d'*x3 + e'*y3 + f'*z3)

# ----------- dynamic programming backward alpha: -----------
# a[(dg+eg+fg)+(dh+eh+fh)+(di+ei+fi)] + b[(dg+eg+fg)+(dh+eh+fh)+(di+ei+fi)] + c[(dg+eg+fg)+(dh+eh+fh)+(di+ei+fi)]
# a[d(g+h+i)+e(g+h+i)+f(g+h+i)] + b[d(g+h+i)+e(g+h+i)+f(g+h+i)] + c[d(g+h+i)+e(g+h+i)+f(g+h+i)]

#                  o1                    o2               o3
# s1  a' = a[d'*x1+e'*x2+f'*x3]  d' = d(x1*g+x2*h+x3*i)   g
# s2  b' = b[d'*y1+e'*y2+f'*y3]  e' = e(y1*g+y2*h+y3*i)   j
# s3  c' = c[d'*z1+e'*z2+f'*z3]  f' = f(z1*g+z2*h+z3*i)   i

# beta(s=1, t=2) - 由第step 2開始，state s1出發，到step結束，可能的機率和 => d'
#                  o1                                o2                  o3
# s1  a' = a[x1*d*d'+x2*e*e'+x3*f*f']  d' = x1*g*g'+x2*h*h'+x3*i*i'   g' = 1
# s2  b' = b[y1*d*d'+y2*e*e'+y3*f*f']  e' = y1*g*g'+y2*h*h'+y3*i*i'   h' = 1
# s3  c' = c[z1*d*d'+z2*e*e'+z3*f*f']  f' = z1*g*g'+z2*h*h'+z3*i*i'   i' = 1

import numpy as np
from HMM.hmm import HiddenMarkovModel


class Evalutor:
    @staticmethod
    def forward_eval_dp(hmm, X, type="sum"):
        # -----
        # type = sum or max
        # -----
        assert type == "sum" or type == "max", "type must equal sum or max"

        # 全部state的index
        S_index = [i for i in range(len(hmm.S))]
        # 觀察到的output的index
        O_index = [hmm.O.index(i) for i in X]
        # gp matrix 為 mxn 的矩陣
        m = len(S_index)  # m = > state 數量
        n = len(X)  # n = > 走了幾步/觀測到多少個output
        dp = np.zeros((m, n))  # dp矩陣

        # for type max
        path_trace = np.zeros((hmm.S_len,n))

        # init
        for i in range(m):
            # 初始化dp矩陣 ex 觀測到 ['2','1','3'] => index 為 [1,0,2]
            #        o1 o0 o2                            o1           o0 o2
            # dp = [[0, 0, 0],          dp = s0 [[in1*(s0吐出o1的機率), 0, 0],
            #       [0, 0, 0],    =>         s1  [in2*(s1吐出o1的機率), 0, 0],
            #       [0, 0, 0]]               s2  [in3*(s2吐出o1的機率), 0, 0]]
            dp[i][0] = hmm.B[S_index[i]][O_index[0]]*hmm.Pi[i]  # si吐出第一個output的機率*初始到si的機率

        # dp
        for j in range(1, n):  # dp 中的 每一個 column => 每一步
            for i in range(m):  # dp 中的 每一個 row => 每個state
                # prior => 前面每個state的機率(dp[:, j-1])*每個state轉換到state i的機率(hmm.transition_matrix[:, i])
                # ex g' = g(d'*x1 + e'*y1 + f'*z1)
                # np.sum(dp[:, j-1] => [d', e', f']
                # self.hmm.transition_matrix[:, i]) => [x1, y1, z1)
                prior = 1
                if type == "sum":
                    prior = np.sum(dp[:, j-1] * hmm.A[:, i])
                elif type == "max":
                    prior = np.max(dp[:, j - 1] * hmm.A[:, i])
                    path_trace[i][j] = int(np.argmax(dp[:, j - 1] * hmm.A[:, i]))

                # op => 該state output觀測到的值的機率 ex g
                op = hmm.B[S_index[i]][O_index[j]]
                # g' = op * prior
                dp[i][j] = op * prior
        if type == "sum":
            return dp
        elif type == "max":
            return dp, path_trace

    @staticmethod
    def backword_eval_dp(hmm, X):
        # 全部state的index
        S_index = [i for i in range(len(hmm.S))]
        # 觀察到的output的index
        O_index = [hmm.O.index(i) for i in X]
        # gp matrix 為 mxn 的矩陣
        m = hmm.S_len  # m = > state 數量
        n = len(X)  # n = > 走了幾步/觀測到多少個output
        dp = np.zeros((m, n))  # dp矩陣

        # init
        for i in range(m):
            dp[i][-1] = 1

        # dp
        for j in range(0, n-1)[::-1]:  # dp 中的 每一個 column => 每一步
            for i in range(m):  # dp 中的 每一個 row => 每個state
                dp[i][j] = np.sum(dp[:, j + 1] * hmm.B[:, O_index[j + 1]] * hmm.A[i, :])
        return dp

    @staticmethod
    def evaluation(hmm, dp, type="forward"):
        # -----
        # 看到一個觀察序列,但看不到state,找出所有可能的路徑的機率的總和
        # -----
        if type == "forward":
            return np.sum(dp[:, -1])
        else:
            O_index = [hmm.O.index(i) for i in X]
            return np.sum(hmm.Pi * hmm.B[:, O_index[0]] * dp[:, 0])

    @staticmethod
    def decoding(hmm, dp, path_trace):
        # -----
        # Viterbi algorithm
        # 看到一個觀察序列 o₁ o₂ ...... oT ，但是看不到狀態序列 s₁ s₂ ...... sT 的情況下，從所有可能的路徑當中，找出機率最大的一條路徑，以及其機率。
        # path_trace[i][j] 的值是 : 第j個step的第i個state，前一個機率最大的state的index
        # dp[i][j] : max forward => 由開始到第j個step的第i個state為止，可以吐出觀測到的observe value的路徑中的最大機率
        # -----
        p = -1e9  # 最大機率路徑的機率
        m,n = np.shape(path_trace) # m => state num, n => step num
        path = [-1 for i in range(n)] # 最大機率的路徑

        # init
        p = np.max(dp[:,-1])  # 由dp的最後一個stept，找到最大機率的state => 最大機率路徑的機率
        path[-1] = int(np.argmax(dp[:,-1]))  # 路徑的最後一個state

        for i in range(1,n)[::-1]:  # 由後往前追蹤機率最到的state
            # 第i-1個state = path_trace[前一個step機率最大的state的index][第i個step]
            path[i-1] = int(path_trace[path[i]][i])

        return p, [hmm.S[i] for i in path]
        
if __name__=="__main__":
    # HMM
    # 狀態 開心
    S = ['開心', '難過']
    O = ['唱歌', '跳舞', '哭泣']
    A = [[0.3, 0.7],
         [0.6, 0.4]]
    B = [[0.4, 0.5, 0.1],
         [0.2, 0.1, 0.7]]
    Pi = [0.5,0.5]

    test_hmm = HiddenMarkovModel(S=S, O=O, A=A, B=B, Pi=Pi)
    S_path, O_path = test_hmm.random_walk_n_times(20)
    print("hmm : ")
    print("S_path : {}".format(S_path))
    print("O_path : {}".format(O_path))

    # forward
    print("\n-------------- evaluate --------------")
    X = ['唱歌', '跳舞', '哭泣']
    eval = Evalutor()
    f_dp = eval.forward_eval_dp(test_hmm, X)
    print("--- forward dp : ---")
    print(f_dp)
    p1 = eval.evaluation(test_hmm, f_dp, "forward")
    print("--- all_path_probability : ---")
    print(p1)
    print("-----------------------------------")
    # backward
    X = ['唱歌', '跳舞', '哭泣']
    b_dp = eval.backword_eval_dp(test_hmm, X)
    print("--- backward dp : ---")
    print(b_dp)
    p2 = eval.evaluation(test_hmm, b_dp, "backward")
    print("--- all_path_probability : ---")
    print(p2)

    # max
    print("\n-------------- decoding --------------")
    X = ['跳舞', '哭泣', '哭泣']
    eval = Evalutor()
    f_dp, path_trace = eval.forward_eval_dp(test_hmm, X, "max")
    print("forward dp max:")
    print(f_dp)
    p3, path3 = eval.decoding(test_hmm, f_dp, path_trace)
    print("max path probability :")
    print(p3)
    print("max probability path:")
    print(path3)
