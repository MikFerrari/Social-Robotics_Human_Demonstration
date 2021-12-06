import numpy as np
from tqdm import tqdm

############# HELPER FUNCTIONS #################################################

def ind2sub(shape,s):
    return s[0]*shape + s[1]

def sub2ind(shape,i):
    return [i//shape,i%shape]

def action2str(demo):
    #Turn action index into str
    res=[]
    for i in demo:
        if i==0:
            res.append("North")
        elif i==1:
            res.append("West")
        elif i==2:
            res.append("South")
        else :
            res.append("East")

    return res

###################### GRID CLASS ##############################################

class grid:

    def __init__(self,s):
        #Initialization of the grid environment and the q_learning objects and parameters
        self.size = s
        self.value_end = 10
        self.value_danger = -10
        self.value_safe = -1
        self.tab = self.value_safe*np.ones((self.size[0],self.size[1]))#np.zeros((self.size[0],self.size[1]))
        self.start = [0,0]
        self.end = [self.size[0],self.size[1]]
        self.state = self.start
        self.q_table = np.zeros((4,self.size[0]*self.size[1]))
        self.discount=0.99
        self.lr = 0.1
        self.epsilon=0.1



    ## Creation of the maze ##
    def add_end(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_end
            self.end = s

        else:
            print("This position corresponds to the starting point!")


    def add_safe(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_safe
        else:
            print("This position corresponds to the starting point!")

    def add_danger(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_danger
        else:
            print("This position corresponds to the starting point!")



    ## State modification and display ##

    def step(self,o):

        new_s=[self.state[0],self.state[1]]

        if self.start==self.end:
            return self.end,10,True

        if o==0 and new_s[0]-1>=0: #North
            new_s[0]-=1
        elif o==1 and new_s[1]-1>=0: #West
            new_s[1]-=1
        elif o==2 and new_s[0]+1<self.size[0]:#South
            new_s[0]+=1
        elif o==3 and new_s[1]+1<self.size[1]:#East
            new_s[1]+=1


        if new_s==self.state:
            return self.state, 0, False

        else:
            #if proceed:
                #self.state = new_s

            return new_s, self.tab[new_s[0],new_s[1]], new_s==self.end

    # Random reset of the environment, e.g, for the q-learning
    def rand_reset(self):
        self.start=sub2ind(self.size[1],np.random.randint(0,self.size[0]*self.size[1]))
        self.state=[self.start[0],self.start[1]]

    # Reset at a specific position, e.g, for comparaison of noisy-rational demonstration
    def reset(self,start):
        self.start=start
        self.state=start


    def print_env(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (np.array([i,j])==self.state).all():
                    print("@",end="   ")
                elif (np.array([i,j])==self.start).all():
                    print("-",end="   ")
                elif (np.array([i,j])==self.end).all():
                    print("0",end="   ")
                elif self.tab[i,j]==self.value_safe:#0:
                    print("x",end="   ")
                else:
                    print("!",end="   ")

            print("\n")



    ## Q-learning ##

    def update_q(self,a,s,s1,r,done):

        #print(s,s1,a)
        s = ind2sub(self.size[1],s)
        s1 = ind2sub(self.size[1],s1)

        if done:
            td = r - self.q_table[a,s]
            #print("IN")
        else:
            td = r + self.discount*np.max(self.q_table[:,s1]) - self.q_table[a,s]

        self.q_table[a,s] = self.q_table[a,s] + self.lr*td

        return td


    def q_learning(self,limit_step,nb_episode):

        self.q_table = np.zeros((4,self.size[0]*self.size[1]))
        n_step=[]
        err_=[]

        init_lr = self.lr

        for e in tqdm(range(nb_episode+1)):
            k=0
            done=False
            self.rand_reset()

            while k<limit_step and not(done):

                if self.start==self.end:
                    pass

                epsi = np.random.rand(1)[0]
                if epsi < self.epsilon:
                    action = np.random.randint(0,4)
                else:
                    #ind_max = np.where(self.q_table[:,ind2sub(self.size[1],self.state)]==np.max(self.q_table[:,ind2sub(self.size[1],self.state)]))[0]
                    #action = np.random.choice(ind_max)
                    action = np.argmax(self.q_table[:,ind2sub(self.size[1],self.state)])

                [new_state, reward, done] = self.step(action)
                #print(self.state,new_state,action)
                err = self.update_q(action,self.state,new_state,reward,done)

                self.state = new_state
                k+=1
                err_.append(abs(err))

            n_step.append(k)


        return n_step, err_
