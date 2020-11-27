# em
# Maximum Likelihood Estimation

# init probability :
# 更新成全部路徑中，由該state i開始的機率

# translation matrix:
# 每個state i 轉換到state j的機率，更新為 (全部有路徑中由state i轉道state j的機率)/(全部路徑中經過state i的機率)

# observe matrix:
# state j output k 的機率為，(全部路徑中 state j output 出 k 的機率)/(全部路徑中經過state j的機率)

import numpy as np
from HMM.hmm import HiddenMarkovModel
from HMM.evaluate import Evalutor

class HMM_Trainer(Evalutor):
    def __init__(self, hmm):
        self.hmm = hmm

    def pass_state_metric(self,f_dp,b_dp,T):
        # ---------
        # gamma[i][j] : 在 step j 穿過 state i 的所有路徑的機率
        # ---------
        gamma = np.zeros((self.hmm.S_len,T))
        for t in range(T):
            p = np.sum(f_dp[:,t] * b_dp[:,t])
            assert p > 0, "p 必須大於0 !!!"
            for i in range(self.hmm.S_len):
                gamma[i][t] = f_dp[i][t] * b_dp[i][t]/p
        return gamma

    def pass_edge_metric(self,f_dp,b_dp,T,O):
        epsilon = np.zeros((self.hmm.S_len,self.hmm.S_len,T))
        for t in range(T-1):
            p = 0
            for i in range(self.hmm.S_len):
                for j in range(self.hmm.S_len):
                    # f_dp[i][t] => S
                    # self.hmm.transition_matrix => SxS
                    # self.hmm.observe_probability => S
                    # b_dp[j][t + 1] => S
                    p += f_dp[i][t] * self.hmm.A[i][j] * self.hmm.B[j][O[t+1]] * b_dp[j][t + 1]
            assert p > 0, "p 必須大於0 !!!"

            for i in range(self.hmm.S_len):
                for j in range(self.hmm.S_len):
                    epsilon[i][j][t] = f_dp[i][t] * self.hmm.A[i][j] * self.hmm.B[j][O[t+1]] * b_dp[j][t + 1] / p

        return epsilon

    def train(self, data):
        O_index = [self.hmm.O.index(i) for i in data]
        T = len(data)
        # get forward dp
        f_dp = self.forward_eval_dp(self.hmm, data)
        # get backward dp
        b_dp = self.backword_eval_dp(self.hmm, data)
        # get gamma
        gamma = self.pass_state_metric(f_dp,b_dp,T)
        # get epsilon
        epsilon = self.pass_edge_metric(f_dp,b_dp,T,O_index)

        # 更新 init probability
        for i in range(self.hmm.S_len):
            self.hmm.Pi[i] = gamma[i][0]

        # 更新 transition_matrix
        for i in range(self.hmm.S_len):
            p2 = 0
            for t in range(T):
                p2 += gamma[i][t]
            assert p2 > 0, "p2 必須大於0 !!!"
            for j in range(self.hmm.S_len):
                p1 = 0
                for t in range(T):
                    p1 += epsilon[i][j][t]
                self.hmm.A[i][j] = p1/p2

        # 更新 observe probability
        for i in range(self.hmm.S_len):
            p = np.zeros(self.hmm.O_len)
            p2=0
            for t in range(T):
                p[O_index[t]] += gamma[i][t]
                p2 += gamma[i][t]
            assert p2 > 0, "p2 必須大於0 !!!"
            for k in range(self.hmm.O_len):
                self.hmm.B[i][k] = p[k] / p2



if __name__ =="__main__":
    # ------------ Target HMM ------------
    S = ['開心', '難過']
    O = ['唱歌', '跳舞', '哭泣']
    Pi = [0.8, 0.2]
    A = [[0.3, 0.7],
         [0.6, 0.4]]
    B = [[0.4, 0.5, 0.1],
         [0.2, 0.1, 0.7]]

    target_hmm = HiddenMarkovModel(S=S, O=O, A=A, B=B, Pi=Pi)

    # ------------ Test HMM ------------
    S = ['開心', '難過']
    O = ['唱歌', '跳舞', '哭泣']

    test_hmm = HiddenMarkovModel(S=S, O=O)

    # init trainer
    trainer = HMM_Trainer(test_hmm)
    # train
    for i in range(50):
        S_path, O_path = target_hmm.random_walk_n_times(100)
        trainer.train(O_path)

    print("---------- After train -------------")
    print("Pi :")
    print(test_hmm.Pi)
    print("Translate matric :")
    print(test_hmm.A)
    print("Observe matric :")
    print(test_hmm.B)
