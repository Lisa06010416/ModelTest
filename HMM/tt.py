import numpy as np
class MarkovModel():
    def __init__(self, S):
        # S = {s1, s2, ...}
        self.S = S
        self.S_len = len(S)
        self.S_index = [i for i in  range(self.S_len)]
        self.transition_matrix = self.init_probability(self.S_len, self.S_len)
        self.s0_probability = self.init_probability(self.S_len)


    # 把值初始化為1/self.S_len => 一開始的機率相同
    def init_probability(self, *size):
        assert type(size) == tuple, "請輸入初始化的維度tuple ex (3,3)"
        for i in size:
            if type(i) != int:
                assert type(size) == tuple, "維度tuple只能含有數字"
        return np.full(size, 1/self.S_len)

    # 根據transition_matrix隨機走travel_num步 => 直接走Ｔ不
    def random_walk(self, travel_num):
        path = []

        # start
        st_index = np.random.choice(self.S_index,p=self.s0_probability.ravel())
        path.append(self.S[st_index])

        # 走travel_num步
        for t in range(travel_num):
            st_index = np.random.choice(self.S_index, p=self.transition_matrix[st_index].ravel())
            path.append(self.S[st_index])

        return path

    # 根據當前state走一部


class HiddenMarkovModel(MarkovModel):
    def __init__(self, S, V):
        MarkovModel.__init__(self, S)
        self.V = V
        self.V_len = len(self.V)
        self.V_index = [i for i in range(len(V))]
        self.observe_probability = self.init_probability(self.S_len, self.V_len)

    # 走一部與走Ｔ不
    def observe_random_walk(self, travel_num):
        path = []

        # start
        st_index = np.random.choice(self.S_index, p=self.s0_probability.ravel())
        path.append(self.S[st_index])

        # 走travel_num步
        for t in range(travel_num):
            st_index = np.random.choice(self.S_index, p=self.transition_matrix[st_index].ravel())
            path.append(self.S[st_index])

        return path

# simple random work and init transition_matrix
S = ['a', 'b', 'c']
MM = MarkovModel(S)
path = MM.random_walk(3)
print(path)