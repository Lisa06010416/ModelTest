import numpy as np
import random

class MarkovModel():
    def __init__(self, S=None, A=None, Pi=None):
        # atate S = {s1, s2, ...}
        self.S = S
        self.S_len = len(S)

        # init init-probability pi
        if not Pi:
            self.Pi = np.array(self.init_probability(self.S_len))
        else:
            self.Pi = np.array(Pi)

        # init transition_matrix A
        A = np.array(A)
        if not A.any():
            self.A = np.array(self.init_probability(self.S_len, self.S_len))
        else:
            self.A = np.array(A)


    # 把值初始化為1/self.S_len => 一開始的機率相同
    def init_probability(self, *size):
        assert type(size) == tuple, "請輸入初始化的維度tuple ex (3,3)"
        for i in size:
            if type(i) != int:
                assert type(size) == tuple, "維度tuple只能含有數字"

        if len(size) == 2:
            m, n = size
            metric = np.zeros(size)
            for i in range(m):
                randomlist = [random.randint(0, 100) for _ in range(n)]
                Sum = sum(randomlist)
                for j in range(n):
                    metric[i][j] = randomlist[j] / Sum
        else :
            n = size[0]
            metric = np.zeros(size)
            randomlist = [random.randint(0, 100) for _ in range(n)]
            Sum = sum(randomlist)
            for j in range(n):
                metric[j] = randomlist[j] / Sum

        return metric

    # 根據transition_matrix隨機走travel_num步
    def random_walk_n_times(self, n):
        # 走travel_num步
        path = []
        state_now = self.random_work_start()
        path.append(state_now)
        for _ in range(n):
            state_now, path = self.random_walk(state_now, path)
        return path

    # start時
    def random_work_start(self):
        st = np.random.choice(self.S, p=self.Pi.ravel())
        return st # next state

    # 由某個state到其他state
    def random_walk_state(self, state):
        pre_st_index = self.S.index(state)
        st = np.random.choice(self.S, p=self.A[pre_st_index].ravel())
        return st # next state

    # 根據當前state走一步
    def random_walk(self, state_now, path):
        if not state_now:
            state_now = self.random_work_start()
        else:
            state_now = self.random_walk_state(state_now)
        path.append(state_now)
        return state_now, path


class HiddenMarkovModel(MarkovModel):
    def __init__(self,
                 S=None,
                 O=None,
                 A=None,
                 B=None,
                 Pi=None):

        MarkovModel.__init__(self, S=S,A=A,Pi=Pi)
        self.O = O
        self.O_len = len(self.O)

        # init observe_probability B
        B = np.array(B)
        if not B.any():
            self.B = np.array(self.init_probability(self.S_len, self.O_len)) # 每個state有機率吐出V1~Vn的機率
        else:
            self.B = np.array(B)


    # 走每一步根據機率observe_probability吐出一個v(屬於V)
    def output_o(self, state):
        st_index = self.S.index(state)
        v = np.random.choice(self.O, p=self.B[st_index].ravel())
        return v

    # 根據transition_matrix隨機走travel_num步
    def random_walk_n_times(self, n):
        S_path = [] # 每個time step的state
        O_path = [] # 每個time step的observe value
        # start
        state_now = self.random_work_start()
        S_path.append(state_now)
        O_path.append(self.output_o(state_now))
        # change state
        for _ in range(n-1):
            state_now, S_path = self.random_walk(state_now, S_path)
            O_path.append(self.output_o(state_now))
        return S_path, O_path




if __name__=="__main__":
    # MM
    # 狀態 開心
    S = ['開心', '難過']
    O = ['唱歌', '跳舞', '哭泣']
    A = [[0.3, 0.7],
         [0.6, 0.4]]
    B = [[0.4, 0.5, 0.1],
         [0.2, 0.1, 0.7]]
    Pi = [1,0]

    hmm = HiddenMarkovModel(S=S, O=O, A=A, B=B, Pi=Pi)
    S_path, O_path = hmm.random_walk_n_times(5)
    print("S_path : {}".format(S_path))
    print("O_path : {}".format(O_path))

