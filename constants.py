import math
import numpy as np

# Data size scales
BYTE = 8
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB
PB = 1024*TB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3


slot = 0.1 # time slot

local_f = 1*GHZ  # local CPU frequency

# Tasks
Prtask = 0.4  # Probability of task coming at any slot
data_size = np.array([10,1,8,0.1])*KB  # kB Unified distributed
cycle_number = 7.375e2

width = 1*MHZ # NW bandwidth
noise = 1.5e-4 

MECS_x = [10,-10,20] # MEC location : x-coordinate
MECS_y = [10,10,0]  # MEC location : y-coordinate
P_send = 0.5

user_X = 0  # user location : x-coordinate
user_Y = 10  # user location : y-coordinate

