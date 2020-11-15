import numpy as np

class MarkovModel():
    def __init__(self, S, transition_matrix=np.array([]), s0_probability=None):
        # S = {s1, s2, ...}
        self.S = S
        self.S_len = len(S)

        # init transition_matrix
        if not transition_matrix.any():
            self.transition_matrix = self.init_probability(self.S_len, self.S_len)
        else:
            self.transition_matrix = transition_matrix

        # init s0_probability
        if not s0_probability:
            self.s0_probability = self.init_probability(self.S_len)
        else:
            self.s0_probability = s0_probability

        self.state_now = None
        self.start_probibality = None
        self.path = [] # record state has visited


    # 把值初始化為1/self.S_len => 一開始的機率相同
    def init_probability(self, *size):
        assert type(size) == tuple, "請輸入初始化的維度tuple ex (3,3)"
        for i in size:
            if type(i) != int:
                assert type(size) == tuple, "維度tuple只能含有數字"
        return np.full(size, 1/self.S_len)

    # 根據transition_matrix隨機走travel_num步
    def random_walk_n_times(self, n):
        # 走travel_num步
        for _ in range(n):
            self.random_walk()

    # start時
    def random_work_start(self):
        st = np.random.choice(self.S, p=self.s0_probability.ravel())
        return st # next state

    # 由某個state到其他state
    def random_walk_state(self, state):
        pre_st_index = self.S.index(state)
        st = np.random.choice(self.S, p=self.transition_matrix[pre_st_index].ravel())
        return st # next state

    # 根據當前state走一步
    def random_walk(self):
        if not self.state_now:
            self.state_now = self.random_work_start()
        else:
            self.state_now = self.random_walk_state(self.state_now)
        self.path.append(self.state_now)


class HiddenMarkovModel(MarkovModel):
    def __init__(self,
                 S,
                 V,
                 transition_matrix=np.array([]),
                 s0_probability=None,
                 observe_probability=np.array([])):

        MarkovModel.__init__(self, S,transition_matrix,s0_probability)
        self.V = V
        self.V_len = len(self.V)
        self.V_path = []

        # init observe_probability
        if not observe_probability.any():
            self.observe_probability = self.init_probability(self.S_len, self.V_len) # 每個state有機率吐出V1~Vn的機率
        else:
            self.observe_probability = observe_probability

    # 走每一步根據機率observe_probability吐出一個v(屬於V)
    def get_v(self, state):
        st_index = self.S.index(state)
        v = np.random.choice(self.V, p=self.observe_probability[st_index].ravel())
        return v

    def random_walk(self):
        if not self.state_now:
            self.state_now = self.random_work_start()
        else:
            self.state_now = self.random_walk_state(self.state_now)
        self.path.append(self.state_now)
        self.V_path.append(self.get_v(self.state_now))


# MM
S = ['a', 'b', 'c']
transition_matrix = np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])

MM = MarkovModel(S=S, transition_matrix=transition_matrix)
MM.random_walk_n_times(3)
print("MM :")
print(MM.path)

# HMM
V = ['1','2','3']
observe_probability = np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
HMM = HiddenMarkovModel(S=S, V=V, transition_matrix=transition_matrix, observe_probability=observe_probability)
HMM.random_walk_n_times(3)
print("HMM :")
print(HMM.path)
print(HMM.V_path)
