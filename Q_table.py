import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.6, e_greedy=0.9999): # 학습 전 초기화
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation): # Q 테이블에 존재하는 Q 값 중 (e-greedy에 따라) 가장 큰 Q 값 가지는 action 택함
        self.check_state_exist(observation) # q table에 존재하는지 먼저 확인
        # action selection
        if np.random.uniform() > self.epsilon: # 탐욕
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            flag = 1
        else: # 탐색
            # choose random action
            action = np.random.choice(self.actions)
            self.epsilon = self.epsilon*0.9999
            flag = 0
        return action,flag,self.epsilon # 선택힌 action, 탐색/탐욕 중 어떤 것 했는지, epsilon 값

    def learn(self, s, a, r, s_): # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)] 으로 Q 값 갱신
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        
    def check_state_exist(self, state): # q table에 값 존재 확인
        if state not in self.q_table.index:
            # append new state to q table using concat
            new_state = pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state,
        )
            self.q_table = pd.concat([self.q_table, new_state.to_frame().T])

'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
'''         