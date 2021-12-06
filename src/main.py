import numpy as np
import matplotlib.pyplot as plt # for plotting of the dependance on temperature
import grid as gd


################## HELPER FUNCTIONS ##################
def compute_prob_variability(mdp,temp):
    prob_table = []
    std_table = []
    for i in range(mdp.size[0]):
        for j in range(mdp.size[1]):
            probs = mdp.softmax_distribution([i,j],temp)
            prob_table.append(probs)
            std_table.append(np.std(np.array(probs)))

    return prob_table, std_table


def plot_std_histogram(temperature_values, std_values, grid_names, i):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(np.arange(0,len(temperature_values)),std_values)
    ax.set_ylabel('Probability Standard Deviation')
    ax.set_xlabel('Temperature Value')
    ax.set_title('Randomness of the probability distribution varying the temperature')
    ax.set_xticks(np.arange(0,len(temperature_values)))
    ax.set_xticklabels(tuple([str(x) for x in temperature_values]))
    plt.draw()
    fig.savefig('plots\probVStemp_'+grid_names[i]+'.png', bbox_inches='tight')


if __name__ == "__main__":
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
            # Create environment
            mdp = gd.Grid(size, value_goal, value_danger, value_safe, start_state, goal_state, temp, discount_factor, learning_rate, grid_name=grid_names[i])
            mdp.add_end(goal_state)

            for danger in item:
                mdp.add_danger(danger)
                
            # print(mdp.tab)
        
            # Perform Q-learning
            mdp.q_learning(max_steps, n_episodes, algo, epsilon, temp)  

            ## DEBUG ##
            # mdp.print_env(start_state)

            # policy_optimal, actions_optimal, score_optimal = mdp.reconstruct_policy_from_q(start_state,"optimal",temp)
            # policy_human, actions_human, score_human = mdp.reconstruct_policy_from_q(start_state,"human",temp)

            # action_map_optimal = mdp.get_policy_from_all_states("optimal",temp)
            # action_map_human = mdp.get_policy_from_all_states("human",temp)
            ## END DEBUG ##

            # Print the map of the chosen actions for each state --> generalised policy
            mdp.visualize_all_policies(temp)

            ## DEBUG ##
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
            ## END DEBUG ##
        
            mdp_list.append(mdp)

            # Study the variation of the policy according to different values of temperature
            prob_table, std_table = compute_prob_variability(mdp, temp)

            std_values.append(np.mean(np.array(std_table)))
            softmax_probabilities.append(prob_table)

        plot_std_histogram(temperature_values, std_values, grid_names, i)     

        i += 1
