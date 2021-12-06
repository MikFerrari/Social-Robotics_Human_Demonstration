#!/usr/bin/env python
# coding: utf-8

# # Introduction to Generative Models of a Demonstrator’s Actions
# ## Practical Work Session #3
# ### Michele Ferrari
# ### 06/12/2021
# 
# ![gridworlds.png](attachment:gridworlds.png)
# 
# ## Table of Contents
# 1. MDP Development: Gridworld Environment
# 2. Boltzmann Model Implementation
# 3. Reinforcement Learning Algorithm
# 4. Simulations
# 5. Temperature $\tau$
# 6. LESS is More Approach

# ## 1. MDP Development: Gridworld Environment

# ### 1.1 Definition of the Grid class and _Q-learning_ function

# In[1]:


import numpy as np
from tqdm import tqdm
from copy import copy
from PIL import Image, ImageDraw  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt # for plotting of the dependance on temperature

############# HELPER FUNCTIONS #################################################

def ind2sub(n_cols,s):
    return s[0]*n_cols + s[1]

def sub2ind(n_cols,i):
    return [i//n_cols,i%n_cols]

def action2str(demo):
    #Turn action index into str
    res = []
    for i in demo:
        if i == 0:
            res.append("North")
        elif i == 1:
            res.append("West")
        elif i == 2:
            res.append("South")
        else:
            res.append("East")

    return res


# In[8]:


###################### GRID CLASS ##############################################
AGENT_N = 1     # Agent key in dict
GOAL_N = 2      # Goal key in dict
DANGER_N = 3    # Danger key in dict
SAFE_N = 4      # Safe key in dict
START_N = 5     # Start key in dict

class grid:

    ## Initialization of the grid environment and the q_learning objects and parameters ##
    def __init__(self, s, val_end, val_danger, val_safe, start_pos, end_pos, temp, disc=0.99, learn_rate=0.1,grid_name="ooo"):
        self.size = s
        self.value_end = val_end
        self.value_danger = val_danger
        self.value_safe = val_safe
        self.tab = self.value_safe*np.ones((self.size[0],self.size[1]))
        self.start = start_pos
        self.end = end_pos
        self.danger = []
        self.state = self.start
        self.q_table = np.zeros((4,self.size[0]*self.size[1]))
        self.discount = disc
        self.lr = learn_rate
        self.grid_name = grid_name

        self.temperature = temp

        # Visualization
        self.colors = {1: (255, 175, 0),  # blueish color: agent
                       2: (0, 255, 0),    # green color: goal
                       3: (0, 0, 255),    # red: danger
                       4: (0, 0, 0),      # white: safe
                       5: (0, 255, 255)}  # yellow: start

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fwidth = self.size[1]*100
        fheight = self.size[0]*100
        self.out = cv2.VideoWriter('videos\simulation_learning_agent_'+grid_name+'_'+str(self.temperature)+'.mp4', fourcc, 10.0, (fwidth,fheight))

    ## Creation of the maze ##
    def add_end(self,s):
        if not(s == self.start):
            self.tab[s[0],s[1]] = self.value_end
            self.end = s

        else:
            print("This position corresponds to the starting point!")


    def add_safe(self,s):
        if not(s == self.start):
            self.tab[s[0],s[1]] = self.value_safe
        else:
            print("This position corresponds to the starting point!")

    def add_danger(self,s):
        if not(s == self.start):
            self.tab[s[0],s[1]] = self.value_danger
            self.danger.append(s)
        else:
            print("This position corresponds to the starting point!")


    ## State modification and display ##
    def step(self,o):
        new_s = [self.state[0],self.state[1]]

        if self.start == self.end:
            return self.end, self.value_end, True

        if o == 0 and new_s[0]-1 >= 0: #North
            new_s[0] -= 1
        elif o == 1 and new_s[1]-1 >= 0: #West
            new_s[1] -= 1
        elif o == 2 and new_s[0]+1 < self.size[0]: #South
            new_s[0] += 1
        elif o == 3 and new_s[1]+1 < self.size[1]: #East
            new_s[1] += 1

        if new_s == self.state:
            return self.state, 0, False
        else:
            return new_s, self.tab[new_s[0],new_s[1]], new_s == self.end


    # Random reset of the environment, e.g, for the q-learning
    def rand_reset(self):
        self.start = sub2ind(self.size[1],np.random.randint(0,self.size[0]*self.size[1]))
        self.state = [self.start[0],self.start[1]]

        
    # Reset at a specific position, e.g, for comparaison of noisy-rational demonstration
    def reset(self,start,goal):
        self.start = start
        self.end = goal
        self.state = start
   
    # Print the grid in the command shell
    def print_env(self,start_state):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (np.array([i,j]) == self.state).all():
                    print("@", end = "   ")
                elif (np.array([i,j]) == start_state).all():
                    print("-", end = "   ")
                elif (np.array([i,j]) == self.end).all():
                    print("0", end = "   ")
                elif self.tab[i,j] == self.value_safe:
                    print("x", end = "   ")
                else:
                    print("!", end = "   ")

            print("\n")


    def visualize(self):
        env = 255*np.ones((self.size[0], self.size[1], 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.end[0]][self.end[1]] = self.colors[GOAL_N]  # sets the goal location tile to green color
        env[self.start[0]][self.start[1]] = self.colors[START_N]  # sets the start location tile to yellow color
        env[self.state[0]][self.state[1]] = self.colors[AGENT_N]  # sets the agent tile to blue
        if self.danger:
            for i in range(len(self.danger)):
                env[self.danger[i][0]][self.danger[i][1]] = self.colors[DANGER_N]  # sets the danger location to red
        
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even though color definitions are bgr
        img = img.resize((self.size[1]*100, self.size[0]*100), resample=Image.NEAREST)

        draw = ImageDraw.Draw(img)
        for i in range(self.size[0]):
            draw.line((0,100*i, self.size[1]*100,100*i), fill=0)
        for i in range(self.size[1]):
            draw.line((100*i,0, 100*i,self.size[0]*100), fill=0)

        cv2.imshow("image", np.array(img))  # show it!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        self.out.write(np.array(img))

    
    def visualize_all_policies(self, temperature):
        env = 255*np.ones((self.size[0], self.size[1], 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.end[0]][self.end[1]] = self.colors[GOAL_N]  # sets the goal location tile to green color
        if self.danger:
            for i in range(len(self.danger)):
                env[self.danger[i][0]][self.danger[i][1]] = self.colors[DANGER_N]  # sets the danger location to red
        
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even though color definitions are bgr
        img = img.resize((self.size[1]*100, self.size[0]*100), resample=Image.NEAREST)

        draw = ImageDraw.Draw(img)
        for i in range(self.size[0]):
            draw.line((0,100*i, self.size[1]*100,100*i), fill=0)
        for i in range(self.size[1]):
            draw.line((100*i,0, 100*i,self.size[0]*100), fill=0)

        action_map_optimal = self.get_policy_from_all_states("optimal",temperature)
        action_map_human = self.get_policy_from_all_states("human",temperature)

        color_optimal = (0,0,0)
        color_human = (190,95,0)
        for i in range(self.size[0]*self.size[1]):
            # The usual indexing condition (n_row,n_col) has to be swapped due to image representation convention
            start_point = [x*100 for x in sub2ind(self.size[0],i)]
            start_point = [sum(x) for x in zip(start_point, [100/3,100/3])]

            idx = i//(self.size[1]-1) + i%(self.size[1]-1)*self.size[1]
            
            # Avoid plotting actions when in terminal state (goal has already been reached!)
            if sub2ind(self.size[1],idx) == self.end:
                continue

            if action_map_optimal[idx] == "North":
                end_point = [sum(x) for x in zip(start_point, [0,-25])]
            elif action_map_optimal[idx] == "West":
                end_point = [sum(x) for x in zip(start_point, [-25,0])]
            elif action_map_optimal[idx] == "South":
                end_point = [sum(x) for x in zip(start_point, [0,25])]
            elif action_map_optimal[idx] == "East":
                end_point = [sum(x) for x in zip(start_point, [25,0])]

            img = cv2.arrowedLine(np.array(img), [int(x) for x in start_point], \
                                  [int(x) for x in end_point], color_optimal, thickness=2, tipLength = 0.5)

            start_point = [x*100 for x in sub2ind(self.size[0],i)]
            start_point = [sum(x) for x in zip(start_point, [2*100/3,2*100/3])]

            if action_map_human[idx] == "North":
                end_point = [sum(x) for x in zip(start_point, [0,-25])]
            elif action_map_human[idx] == "West":
                end_point = [sum(x) for x in zip(start_point, [-25,0])]
            elif action_map_human[idx] == "South":
                end_point = [sum(x) for x in zip(start_point, [0,25])]
            elif action_map_human[idx] == "East":
                end_point = [sum(x) for x in zip(start_point, [25,0])]

            img = cv2.arrowedLine(np.array(img), [int(x) for x in start_point], \
                                  [int(x) for x in end_point], color_human, thickness=2, tipLength = 0.5)

        cv2.imshow("image", np.array(img))  # show it!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        cv2.imwrite('images\\final_policies_'+self.grid_name+'_'+str(temperature)+'.png', np.array(img))
        cv2.destroyAllWindows()


    ## Q-learning ##
    def update_q(self,a,s,s1,r,done):
        s = ind2sub(self.size[1],s)
        s1 = ind2sub(self.size[1],s1)

        if done:
            td = r - self.q_table[a,s]
        else:
            td = r + self.discount*np.max(self.q_table[:,s1]) - self.q_table[a,s]

        self.q_table[a,s] = self.q_table[a,s] + self.lr*td

        return td
        
   
    def egreedy_policy(self,epsilon):
        e = np.random.uniform(0,1)

        # N.B.: If all elements are equal, choose an action randomly
        if e < epsilon or (self.q_table[:,ind2sub(self.size[1],self.state)] == self.q_table[:,0][0]).all():
            action = np.random.randint(0,4)
        else:
            action = np.argmax(self.q_table[:,ind2sub(self.size[1],self.state)])
            
        return action


    def softmax_distribution(self,s,tau):
        # Returns a soft-max probability distribution over actions
        # Inputs:
        # - Q: a Q-function
        # - s: the state for which we want the soft-max distribution
        # - tau: temperature parameter of the soft-max distribution
        # Output:
        # - action: action selected based on the Boltzmann probability distribution
    
        p = np.zeros(len(self.q_table))
        sum_p = 0
        for i in range(len(p)):
            p[i] = np.exp((self.q_table[i,ind2sub(self.size[1],s)] / tau))
            sum_p += p[i]

        p = p / sum_p

        return p


    def softmax_policy(self,s,tau):
        # Returns a soft-max probability distribution over actions
        # Inputs:
        # - Q: a Q-function
        # - s: the state for which we want the soft-max distribution
        # - tau: temperature parameter of the soft-max distribution
        # Output:
        # - action: action selected based on the Boltzmann probability distribution
        
        p = np.zeros(len(self.q_table))
        sum_p = 0
        for i in range(len(p)):
            p[i] = np.exp((self.q_table[i,ind2sub(self.size[1],s)] / tau))
            sum_p += p[i]

        p = p / sum_p

        # Draw a random action from the distribution
        p_cumsum = np.cumsum(p)
        p_cumsum = np.insert(p_cumsum, 0, 0)

        choice = np.random.uniform(0,1)

        action = int(np.where(choice > p_cumsum)[0][-1])
                  
        # Alternatively:
        # action = int(np.random.choice([0,1,2,3], p=p))
        
        return action
                     
    
    def reconstruct_policy_from_q(self,start_state,optimality_flag,tau):
        policy = []
        actions = []
        state = copy(start_state)
        policy.append(copy(start_state))

        score = 0
        count = 0
        timeout = 2*self.size[0]*self.size[1]
        while state != self.end:

            if optimality_flag == "optimal":
                action = np.argmax(self.q_table[:,ind2sub(self.size[1],state)])
            elif optimality_flag == "human":
                action = self.softmax_policy(state,tau)

            # Prevent agent from being stuck in the same transition forwards and backwards 
            if count >= 2 and state == policy[count-2]:
                while action == actions[count-2]:
                    action = np.random.randint(0,4)

            score += self.q_table[action,ind2sub(self.size[1],state)]

            if action == 0 and state[0]-1 >= 0: #North
                state[0] -= 1
            elif action == 1 and state[1]-1 >= 0: #West
                state[1] -= 1
            elif action == 2 and state[0]+1 < self.size[0]: #South
                state[0] += 1
            elif action == 3 and state[1]+1 < self.size[1]: #East
                state[1] += 1

            policy.append(copy(state))
            actions.append(action)

            # Interrupt path reconstruction if the policy does not converge to the goal
            count += 1
            if count >= timeout:
                break
            
        actions = action2str(actions)

        return policy, actions, score


    def get_policy_from_all_states(self,optimality_flag,tau):
        actions = []
        
        for i in range(self.size[0]*self.size[1]):
            if optimality_flag == "optimal":
                action = np.argmax(self.q_table[:,i])

            elif optimality_flag == "human":
                action = self.softmax_policy(sub2ind(self.size[1],i),tau)

            actions.append(action)
            
        actions = action2str(actions)
        return actions

            
    def q_learning(self,limit_step,nb_episode,algorithm="optimal",epsilon=0.1,tau=1,display_flag=False):

        self.q_table = np.zeros((4,self.size[0]*self.size[1]))
        n_step = []
        global_temporal_differences = []

        for e in tqdm(range(nb_episode)):
            k = 0
            done = False
            self.rand_reset()

            if e % 100 == 0 and display_flag == True:
                self.visualize()
            
            temporal_differences = []
            while k < limit_step and not(done):

                if self.start == self.end:
                    pass

                if algorithm == "egreedy":
                    # Agent chooses next action according to an epsilon-greedy distribution
                    action = self.egreedy_policy(epsilon)                        
                elif algorithm == "softmax":
                    # Agent chooses next action according to a soft-max distribution
                    action = self.softmax_policy(self.state,tau)

                [new_state, reward, done] = self.step(action)
                
                err = self.update_q(action,self.state,new_state,reward,done)

                self.state = new_state
                k += 1
                temporal_differences.append(abs(err))

                if e % 100 == 0 and display_flag == True:
                    self.visualize()

            n_step.append(k)
            global_temporal_differences.append(temporal_differences)

        if display_flag == True:
            self.out.release()
            cv2.destroyAllWindows()

        return n_step, global_temporal_differences


# ### 1.2 Creation and display of the envirnoments

# In[21]:


# Define MDP structure
size = [5,6]
start_state = [2,0] # tile (3,1)
goal_state = [2,5] # tile (3,6)

value_goal = 10
value_safe = 0
value_danger = -2

# Define RL parameters
discount_factor = 0.95
learning_rate = 0.8

# Define Q-learning parameters
max_steps = 100
n_episodes = 10000
algo = "egreedy"
epsilon = 0.2
temperature_values = [0.1,0.25,0.5,1,10,100]

# Define grid: specify dangerous states
grid_ooo = []
grid_oox = [[2,3],[2,4],[3,4],[4,1],[4,2],[4,3],[4,4]]
grid_oxo = [[1,2],[1,3],[1,4],[2,2],[3,2],[3,3]]
grid_xoo = [[0,1],[0,2],[0,3],[0,4],[1,1],[2,1],[3,1]]
grid_oxx = [[1,2],[1,3],[1,4],[2,2],[2,3],[2,4],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]]
grid_xox = [[0,1],[0,2],[0,3],[0,4],[1,1],[2,1],[3,1],[4,1],[4,2],[4,3],[4,4],[3,4],[2,4],[2,3]]
grid_xxo = [[0,1],[0,2],[0,3],[0,4],[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[3,2],[3,3]]
grid_xxx = [[0,1],[0,2],[0,3],[0,4],[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]]

grid_list = [grid_ooo, grid_oox, grid_oxo, grid_xoo, grid_oxx, grid_xox, grid_xxo, grid_xxx]
grid_names = ["ooo", "oox", "oxo", "xoo", "oxx", "xox", "xxo", "xxx"]

mdp_list = []
i = 0
# Creation of the different MDPs
for item in grid_list:
    
    softmax_probabilities = []
    std_values = []

    for temp in temperature_values:
        mdp = grid(size, value_goal, value_danger, value_safe, start_state, goal_state, temp, discount_factor, learning_rate, grid_name=grid_names[i])
        mdp.add_end(goal_state)

        for danger in item:
            mdp.add_danger(danger)
            
        # print(mdp.tab)
    
        mdp.q_learning(max_steps, n_episodes, algo, epsilon, temp, display_flag=False)  

        # mdp.print_env(start_state)

        # policy_optimal, actions_optimal, score_optimal = mdp.reconstruct_policy_from_q(start_state,"optimal",temp)
        # policy_human, actions_human, score_human = mdp.reconstruct_policy_from_q(start_state,"human",temp)

        # action_map_optimal = mdp.get_policy_from_all_states("optimal",temp)
        # action_map_human = mdp.get_policy_from_all_states("human",temp)

        mdp.visualize_all_policies(temp)

        # Find the policy given a specific start state
        # print("Optimal policy: ")
        # print(policy_optimal)
        # print(actions_optimal)
        # print(score_optimal)
        # print("Human policy: ")
        # print(policy_human)
        # print(actions_human)
        # print(score_human)

        # Find the best action for any given state (generalized policy)
        # print("Optimal actions: ")
        # print(action_map_optimal)
        # print("Human actions: ")
        # print(action_map_human)
    
        mdp_list.append(mdp)

        prob_table = []
        std_table = []
        for i in range(mdp.size[0]):
            for j in range(mdp.size[1]):
                probs = mdp.softmax_distribution([i,j],temp)
                prob_table.append(probs)
                std_table.append(np.std(np.array(probs)))

        std_values.append(np.mean(np.array(std_table)))
        softmax_probabilities.append(prob_table)
        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(np.arange(0,len(temperature_values)),std_values)
    ax.set_ylabel('Probability Standard Deviation')
    ax.set_xlabel('Temperature Value')
    ax.set_title('Randomness of the probability distribution varying the temperature')
    ax.set_xticks(np.arange(0,len(temperature_values)))
    ax.set_xticklabels(tuple([str(x) for x in temperature_values]))
    plt.show()
    plt.savefig('plots\probVStemp_'+grid_names[i]+'_'+str(temp)+'.png', bbox_inches='tight')

    i += 1


# ## 2. Boltzmann Model: Role of Temperature Parameter

# In[ ]:





# ## 3. Reinforcement Learning Algorithm

# In[ ]:





# ## Bibliography
# [1] Ho, M., Littman, M., Cushman, F., & Austerweil, J.L. (2018). Effectively Learning from Pedagogical Demonstrations. Cognitive Science.  
# 
# [2] Milli, S., & Dragan, A.D. (2019). Literal or Pedagogic Human ? Analyzing Human Model Misspecification in Objective Learning. UAI.
# 
# [3] Andreea Bobu, Dexter R.R. Scobee, Jaime F. Fisac, S. Shankar Sastry, and Anca D. Dragan. 2020. LESS is More: Rethinking Probabilistic Models of Human Behavior. In Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI ’20), March 23–26, 2020, Cambridge, United Kingdom.
