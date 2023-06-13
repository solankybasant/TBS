#https://www.kaggle.com/code/petarjeroncic/markov-chains-simulation-in-python

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


states = {
    -1 : "Loss",
    0 : "Draw",
    1 : "Win"
}
states


transition = np.array([[0.65, 0.1, 0.25],[0.3,0.5,0.2],[0.35,0.1,0.55]])
transition


n = 15
br=0
points = 0
start_state = 0
print(states[start_state], "-->", end=" ")
prev_state = start_state

while n:
    if(prev_state==0):
        points+=1
    elif(prev_state==1):
        points+=3
    curr_state = np.random.choice([-1,0,1], p =transition[prev_state+1])
    print(states[curr_state], "-->", end=" ")
    prev_state=curr_state
    n-=1
    br+=1
print("stop")
if(prev_state==0):
        points+=1
elif(prev_state==1):
        points+=3
print("Points: ", points)
print("Success: ", (points)/(3*br))


##following is the pi(n) as a function of n. For each n, many sample paths are averaged
num_sample=200
pi_evolution=[]
for n in range(100,600,200):
    start_state = 0
    prev_state = start_state
    num_samples_n=np.array([0,0,0])
    for _ in range(num_sample):
        for i in range(n):
            curr_state = np.random.choice([-1,0,1], p =transition[prev_state+1])
            prev_state=curr_state
        num_samples_n[curr_state]=num_samples_n[curr_state]+1
    pi_evolution.append(num_samples_n/num_sample)

plt.plot(pi_evolution)
plt.show()


##following is the pi(n) as a function of n.
#For each n, averaging is done over one sample path

##
##n = 100
##br = n
##br_l=0
##br_d=0
##br_w=0
##start_state = 0
##prev_state = start_state
##
##states_history=[]
##
##while n:
##    curr_state = np.random.choice([-1,0,1], p =transition[prev_state+1])
##    if curr_state==-1:
##        br_l+=1
##    elif curr_state==0:
##        br_d+=1
##    else:
##        br_w+=1
##    prev_state=curr_state
##    states_history.append(curr_state)
##    n-=1
##print("Loss : ", br_l/br) 
##print("Draw : ", br_d/br)
##print("Win : ", br_w/br)
##plt.plot(states_history)
##plt.show()

##
##
##steps = 10**3
##start_state = 0
##pi = np.array([0,0,0])
##pi[start_state+1]+=1
##prev_state = start_state
##
##pi_history=[]
##for i in range(len(pi)):
##    pi_history.append([])
##
##for i in range(steps):
##    curr_state = np.random.choice([-1,0,1], p=transition[prev_state+1])
##    pi[curr_state+1]+=1
##    prev_state=curr_state
##    pi_temp=pi/(i+1)
##    for j in range(len(pi)):
##        pi_history[j].append(pi_temp[j])
##
##plt.bar(range(steps), np.array(pi_history[0]), color='r')
##plt.bar(range(steps), np.array(pi_history[1]), bottom=np.array(pi_history[0]), color='b')
##plt.bar(range(steps), np.array(pi_history[2]), bottom=np.array(pi_history[1])+np.array(pi_history[0]), color='g')
##plt.show()
##
##print("pi = ",pi/steps)
##
##import scipy.linalg
##values, left = scipy.linalg.eig(transition, right = False, left = True)
##print("left eigen vectors =\n", left, "\n")
##print("eigen values = \n", values)
##
##
##pi = left[:,0]
##pi_normalized = [(x/np.sum(pi)).real for x in pi]
##print("pi = ", pi_normalized)
##
##
##steps = 10**6
##start_state = 0
##pi = transition[start_state+1]
##for i in range(steps):
##    pi=np.dot(pi,transition)
##
##print("pi = ",pi)
##
##
##steps = 10
##transition_n = transition
##for i in range(steps):
##    transition_n=np.matmul(transition_n,transition)
##
##print("Matrix: \n", transition_n, "\n")
##print("pi = ",transition_n[1])
##
##
##steps = 1000
##transition_n = transition
##for i in range(steps):
##    transition_n=np.matmul(transition_n,transition)
##
##print("Matrix: \n", transition_n, "\n")
##print("pi = ",transition_n[1])
##
##
##
##def find_prob(seq, A, pi) :
##    start_state = seq[0]
##    prob = pi[start_state]
##    prev_state = start_state
##    
##    for i in range(1,len(seq)):
##        curr_state = seq[i]
##        prob*=A[prev_state][curr_state]
##        prev_state=curr_state
##    return prob
##
##print(find_prob([1,0,-1,-1,-1,1], transition, pi_normalized))
##
