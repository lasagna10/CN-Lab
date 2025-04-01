from env import Maze
from dqn import DQN
from constants import *
import pdb
import warnings
import torch
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from random import choice
from Q_table import QLearningTable
from func import trans

ACTIONS = pd.Series([], dtype=int) # action 인덱스 저장
for i in range(66):
    ACTIONS = pd.concat([ACTIONS, pd.Series([i])], ignore_index=True)
warnings.filterwarnings("ignore")

r = pd.Series([], dtype=int) # 에피소드별 누적 평균 보상 (DQN)
rl_r = pd.Series([], dtype=int) # 에피소드별 누적 평균 보상 (Q-learning)
e = pd.Series([], dtype=int) # 에피소드 번호
Time = pd.Series([], dtype=int) # 에피소드별 평균 latency (DQN)
rl_Time = pd.Series([], dtype=int) # 에피소드별 평균 latency (Q-learning)
action = torch.Tensor([]) # 에이전트의 액션 저장하는 텐서
RL = QLearningTable(ACTIONS) # Q-learning 에이전트 인스턴스 초기화

epsilon = 0.9995 # e-greedy 초기값
MEMORY_CAPACITY = 1000
#pdb.set_trace()
env = Maze(local_f) # Maze 클래스에서 환경 구현. step( 함수가 보상과 지연시간 반환

agent = DQN() # DQN 에이전트 초기화
sum_ = 0 # DQN 누적 보상 합계
rl_reward_sum = 0 # Q-learning 누적 보상 합계
t_sum = 0 # DQN 누적 지연 시간 합계
rl_sum = 0 # Q-learning 누적 지연 시간 합계

for episode in range(30000): 
    # 에피소드 수 30000
    # "환경 초기화 -> 액션 선택 -> 상태 전이" 반복
    # 보상, latency를 얻어 각각 DQN과 Q-learning 에이전트에 저장, 학습
       
    observation = env.reset()
    action,ep = agent.choose_action(observation) # DQN
    rl_action,rl_flag,rl_ep = RL.choose_action(str(observation)) # Q-Learning
    ep = round(ep,2)
    
    t_action = action

    # DQN 과정에서 정수형 액션 인덱스 -> 실제 액션(action vector) 매핑
    # t_action = [offload ratio, server type, channel state]
    
    if t_action == 65:
        t_action = [0,2,1]
    elif t_action == 64:
        t_action = [0,2,0]
    elif t_action == 63:
        t_action = [0.1,2,1]
    elif t_action == 62:
        t_action = [0.1,2,0]
    elif t_action == 61:
        t_action = [0.2,2,1]
    elif t_action == 60:
        t_action = [0.2,2,0]
    elif t_action == 59:
        t_action = [0.3,2,1]
    elif t_action == 58:
        t_action = [0.3,2,0]
    elif t_action == 57:
        t_action = [0.4,2,1]
    elif t_action == 56:
        t_action = [0.4,2,0]
    elif t_action == 55:
        t_action = [0.5,2,1]
    elif t_action == 54:
        t_action = [0.5,2,0]
    elif t_action == 53:
        t_action = [0.6,2,1]
    elif t_action == 52:
        t_action = [0.6,2,0]
    elif t_action == 51:
        t_action = [0.7,2,1]
    elif t_action == 50:
        t_action = [0.7,2,0]
    elif t_action == 49:
        t_action = [0.8,2,1]
    elif t_action == 48:
        t_action = [0.8,2,0]
    elif t_action == 47:
        t_action = [0.9,2,1]
    elif t_action == 46:
        t_action = [0.9,2,0]
    elif t_action == 45:
        t_action = [1,2,1]
    elif t_action == 44:
        t_action = [1,2,0]
        
    elif t_action == 43:
        t_action = [0,1,1]
    elif t_action == 42:
        t_action = [0,1,0]
    elif t_action == 41:
        t_action = [0,0,1]
    elif t_action == 40:
        t_action = [0,0,0]
    elif t_action == 39:
        t_action = [0.1,1,1]
    elif t_action == 38:
        t_action = [0.1,1,0]
    elif t_action == 37:
        t_action = [0.1,0,1]
    elif t_action == 36:
        t_action = [0.1,0,0]
    elif t_action == 35:
        t_action = [0.2,1,1]
    elif t_action == 34:
        t_action = [0.2,1,0]
    elif t_action == 33:
        t_action = [0.2,0,1]
    elif t_action == 32:
        t_action = [0.2,0,0]
    elif t_action == 31:
        t_action = [0.3,1,1]
    elif t_action == 30:
        t_action = [0.3,1,0]
    elif t_action == 29:
        t_action = [0.3,0,1]
    elif t_action == 28:
        t_action = [0.3,0,0]
    elif t_action == 27:
        t_action = [0.4,1,1]
    elif t_action == 26:
        t_action = [0.4,1,0]
    elif t_action == 25:
        t_action = [0.4,0,1]
    elif t_action == 24:
        t_action = [0.4,0,0]
    elif t_action == 23:
        t_action = [0.5,1,1]
    elif t_action == 22:
        t_action = [0.5,1,0]
    elif t_action == 21:
        t_action = [0.5,0,1]
    elif t_action == 20:
        t_action = [0.5,0,0]
    elif t_action == 19:
        t_action = [0.6,1,1]
    elif t_action == 18:
        t_action = [0.6,1,0]
    elif t_action == 17:
        t_action = [0.6,0,1]
    elif t_action == 16:
        t_action = [0.6,0,0]
    elif t_action == 15:
        t_action = [0.7,1,1]
    elif t_action == 14:
        t_action = [0.7,1,0]
    elif t_action == 13:
        t_action = [0.7,0,1]
    elif t_action == 12:
        t_action = [0.7,0,0]
    elif t_action == 11:
        t_action = [0.8,1,1]
    elif t_action == 10:
        t_action = [0.8,1,0]
    elif t_action == 9:
        t_action = [0.8,0,1]
    elif t_action == 8:
        t_action = [0.8,0,0]
    elif t_action == 7:
        t_action = [0.9,1,1]
    elif t_action == 6:
        t_action = [0.9,1,0]
    elif t_action == 5:
        t_action = [0.9,0,1]
    elif t_action == 4:
        t_action = [0.9,0,0]
    elif t_action == 3:
        t_action = [1,1,1]
    elif t_action == 2:
        t_action = [1,1,0]
    elif t_action == 1:
        t_action = [1,0,1]
    else:
        t_action = [1,0,0]
    
    observation_, reward ,t= env.step(t_action,action) # DQN 방식 으로 얻은 step 함수의 반환값 # observation(next state)와 reward를 수집
    rl_t_action = trans(rl_action) # Q-learning 방식으로 action을 매핑
    u,rl_reward,rl_t = env.step(rl_t_action,rl_action) # Q-learning 방식으로 얻은 step 함수의 반환값 # 선택한 행동을 환경에 적용
    #if  and
    
    print(f"Episode {episode}: reward={reward}, rl_reward={rl_reward}, t={t}, rl_t={rl_t}")
    
    # 보상과 지연 시간을 DQN / Q-learning 방식으로 각각 계산
    if episode != 0 and reward is not None and rl_reward is not None and not np.isnan(reward) and not np.isnan(rl_reward) :
        #print(episode,rl_t)
        sum_ += reward # 누적 보상 (DQN)
        rl_reward_sum += rl_reward # 누적 보상 (Q)
        re = sum_/episode # 에피소드별 평균 보상 (DQN)
        rl_re = rl_reward_sum/episode # 에피소드별 평균 보상 (Q)
        r = pd.concat([r, pd.Series([re])], ignore_index=True) # 각각 에피소드별 평균 보상 값 기록 -> reward 그래프 그릴 때 사용
        rl_r = pd.concat([rl_r, pd.Series([rl_re])], ignore_index=True)
        rl_sum += rl_t # 누적 지연 시간 (Q)
        t_sum += t # 누적 지연 시간 (DQN)
        tt = t_sum/episode # 평균 지연 시간 (DQN)
        rl_tt = rl_sum/episode # 평균 지연 시간 (Q)
        e = pd.concat([e, pd.Series([episode])], ignore_index=True) # 현재 수행 중인 에피소드의 번호 저장
        Time = pd.concat([Time, pd.Series([tt])], ignore_index=True) # 각각 에피소드별 평균 latency 값 기록
        rl_Time = pd.concat([rl_Time, pd.Series([rl_tt])], ignore_index=True)
    else :
        print(f"⚠️ skip - reward or time invalid (reward={reward}, rl_reward={rl_reward})")
        
    # 상태 전이
    agent.store_transition(observation, action, reward, observation_) # DQN이 transition(s,a,r,s')을 experience replay buffer에 저장 -> 버퍼에 일정량 이상 데이터가 모이면 꺼내어 학습에 활용
    RL.learn(str(observation), rl_action, rl_reward, str(observation_)) # Q learning은 Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)] 형태로 별도의 transition 저장 없이 바로 갱신

    observation = observation_ # 상태 전이 완료, s <- s'
    if agent.memory_counter >= MEMORY_CAPACITY: # DQN은 experience replay buffer가 가득 찬 뒤부터 학습 시작 -> 일정량 이상부터 random sampling + gradient update
        agent.learn()

print("len(e):", len(e))
print("len(Time):", len(Time))
print("len(r):", len(r))
print("Time[:5]:", Time[:5])
print("r[:5]:", r[:5])

plt.figure()
plt.plot(e[10:],Time[10:],'b', label='DQN')
plt.plot(e[10:],rl_Time[10:],'r', label='Q-learning')
plt.legend()
plt.title("Latency (after discarding first 10 episodes)")
plt.show()

plt.figure()
plt.plot(e[10:],r[10:],'b', label='DQN')
plt.plot(e[10:],rl_r[10:],'r', label='Q-learning')
plt.legend()
plt.title("Reward (after discarding first 10 episodes)")
plt.show()

plt.figure()
plt.plot(e,Time,'b', label='DQN')
plt.plot(e,rl_Time,'r', label='Q-learning')
plt.legend()
plt.title("Latency (overall)")
plt.show()

plt.figure()
plt.plot(e,r,'b', label='DQN')
plt.plot(e,rl_r,'r', label='Q-learning')
plt.legend()
plt.title("Reward (overall)")
plt.show()

'''
from env import Maze
from dqn import DQN
from constants import *
import pdb
import warnings
import torch
import numpy as np
import operator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from random import choice
# from TD3 import TD3
# from utils import ReplayBuffer
from Q_table import QLearningTable
from func import trans
ACTIONS = []
for i in range(66):
    ACTIONS.append(i)
warnings.filterwarnings("ignore")

r = []
rl_r = []
e = []
Time = []
rl_Time = []
action = torch.Tensor([])
RL = QLearningTable(ACTIONS)

epsilon = 0.9995
MEMORY_CAPACITY = 1000
#pdb.set_trace()
env = Maze(local_f)

agent = DQN()
sum_ = 0
rl_reward_sum = 0
t_sum = 0
rl_sum = 0

for episode in range(30000):
       
    observation = env.reset()
    action,ep = agent.choose_action(observation)
    rl_action,rl_flag,rl_ep = RL.choose_action(str(observation))
    ep = round(ep,2)
    
    t_action = action

    if t_action == 65:
        t_action = [0,2,1]
    elif t_action == 64:
        t_action = [0,2,0]
    elif t_action == 63:
        t_action = [0.1,2,1]
    elif t_action == 62:
        t_action = [0.1,2,0]
    elif t_action == 61:
        t_action = [0.2,2,1]
    elif t_action == 60:
        t_action = [0.2,2,0]
    elif t_action == 59:
        t_action = [0.3,2,1]
    elif t_action == 58:
        t_action = [0.3,2,0]
    elif t_action == 57:
        t_action = [0.4,2,1]
    elif t_action == 56:
        t_action = [0.4,2,0]
    elif t_action == 55:
        t_action = [0.5,2,1]
    elif t_action == 54:
        t_action = [0.5,2,0]
    elif t_action == 53:
        t_action = [0.6,2,1]
    elif t_action == 52:
        t_action = [0.6,2,0]
    elif t_action == 51:
        t_action = [0.7,2,1]
    elif t_action == 50:
        t_action = [0.7,2,0]
    elif t_action == 49:
        t_action = [0.8,2,1]
    elif t_action == 48:
        t_action = [0.8,2,0]
    elif t_action == 47:
        t_action = [0.9,2,1]
    elif t_action == 46:
        t_action = [0.9,2,0]
    elif t_action == 45:
        t_action = [1,2,1]
    elif t_action == 44:
        t_action = [1,2,0]
        
    elif t_action == 43:
        t_action = [0,1,1]
    elif t_action == 42:
        t_action = [0,1,0]
    elif t_action == 41:
        t_action = [0,0,1]
    elif t_action == 40:
        t_action = [0,0,0]
    elif t_action == 39:
        t_action = [0.1,1,1]
    elif t_action == 38:
        t_action = [0.1,1,0]
    elif t_action == 37:
        t_action = [0.1,0,1]
    elif t_action == 36:
        t_action = [0.1,0,0]
    elif t_action == 35:
        t_action = [0.2,1,1]
    elif t_action == 34:
        t_action = [0.2,1,0]
    elif t_action == 33:
        t_action = [0.2,0,1]
    elif t_action == 32:
        t_action = [0.2,0,0]
    elif t_action == 31:
        t_action = [0.3,1,1]
    elif t_action == 30:
        t_action = [0.3,1,0]
    elif t_action == 29:
        t_action = [0.3,0,1]
    elif t_action == 28:
        t_action = [0.3,0,0]
    elif t_action == 27:
        t_action = [0.4,1,1]
    elif t_action == 26:
        t_action = [0.4,1,0]
    elif t_action == 25:
        t_action = [0.4,0,1]
    elif t_action == 24:
        t_action = [0.4,0,0]
    elif t_action == 23:
        t_action = [0.5,1,1]
    elif t_action == 22:
        t_action = [0.5,1,0]
    elif t_action == 21:
        t_action = [0.5,0,1]
    elif t_action == 20:
        t_action = [0.5,0,0]
    elif t_action == 19:
        t_action = [0.6,1,1]
    elif t_action == 18:
        t_action = [0.6,1,0]
    elif t_action == 17:
        t_action = [0.6,0,1]
    elif t_action == 16:
        t_action = [0.6,0,0]
    elif t_action == 15:
        t_action = [0.7,1,1]
    elif t_action == 14:
        t_action = [0.7,1,0]
    elif t_action == 13:
        t_action = [0.7,0,1]
    elif t_action == 12:
        t_action = [0.7,0,0]
    elif t_action == 11:
        t_action = [0.8,1,1]
    elif t_action == 10:
        t_action = [0.8,1,0]
    elif t_action == 9:
        t_action = [0.8,0,1]
    elif t_action == 8:
        t_action = [0.8,0,0]
    elif t_action == 7:
        t_action = [0.9,1,1]
    elif t_action == 6:
        t_action = [0.9,1,0]
    elif t_action == 5:
        t_action = [0.9,0,1]
    elif t_action == 4:
        t_action = [0.9,0,0]
    elif t_action == 3:
        t_action = [1,1,1]
    elif t_action == 2:
        t_action = [1,1,0]
    elif t_action == 1:
        t_action = [1,0,1]
    else:
        t_action = [1,0,0]
    
    observation_, reward ,t= env.step(t_action,action)
    rl_t_action = trans(rl_action)
    u,rl_reward,rl_t = env.step(rl_t_action,rl_action)
    #if  and
    if episode != 0:
        #print(episode,rl_t)
        sum_ += reward
        rl_reward_sum += rl_reward
        re = sum_/episode
        rl_re = rl_reward_sum/episode
        r.append(re)
        rl_r.append(rl_re)
        rl_sum += rl_t
        #print("总的",rl_sum)
        t_sum += t
        tt = t_sum/episode
        rl_tt = rl_sum/episode
        #print("平均",rl_tt)
        e.append(episode)
        Time.append(tt)
        rl_Time.append(rl_tt)
    
    #file_handle=open('ran.txt',mode='a')
    #if agent.flag is 1 and agent.memory_counter >= MEMORY_CAPACITY:
    #file_handle.write("epsilon"+ str(rl_ep)+"       state"+str(observation)+"      action:"+str(rl_action)+"      reward:"+str(rl_reward)+"      t"+str(rl_t)+"\n")
    #if agent.memory_counter < MEMORY_CAPACITY:
     #   print("action:",t_action,"reward:",reward)
    agent.store_transition(observation, action, reward, observation_)
    RL.learn(str(observation), rl_action, rl_reward, str(observation_))

    observation = observation_
    if agent.memory_counter >= MEMORY_CAPACITY:
        agent.learn()
#file_handle.close()

plt.figure()
plt.plot(e[10:],Time[10:],'b', label='DQN')
plt.plot(e[10:],rl_Time[10:],'r', label='Q-learning')
plt.legend()
plt.title("时延（舍弃10个点）")
plt.show()

plt.figure()
plt.plot(e[10:],r[10:],'b', label='DQN')
plt.plot(e[10:],rl_r[10:],'r', label='Q-learning')
plt.legend()
plt.title("reward（舍弃10个点）")
plt.show()

plt.figure()
plt.plot(e,Time,'b', label='DQN')
plt.plot(e,rl_Time,'r', label='Q-learning')
plt.legend()
plt.title("时延")
plt.show()

plt.figure()
plt.plot(e,r,'b', label='DQN')
plt.plot(e,rl_r,'r', label='Q-learning')
plt.legend()
plt.title("reward")
plt.show()
'''