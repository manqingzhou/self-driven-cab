# self-driven-cab
PROBLEM:
We aim to create a self-driving cab, its job is to pick up the passenger at one location and drop them off in another. This cab will also take care of:
Drop off the passenger to the right location.
Save passenger's time by taking minimum time possible to drop off
Take care of passenger's safety and traffic rules

The main idea of this project is to learning from the past and solve the problemfaster
ALGORITHMS:
q_learning function:

def q_learning(episode_limit,q_table,alpha =0.1,gamma = 0.6):
    y = np.zeros((episode_limit,1))
    for i in range(1,episode_limit):
        state = env.reset() #every single time we reset the enviroment
        epoch,penalty,reward =0,0,0
        done = False
   
        while not done:
            action = choose_a(state)
            next_state,reward,done,info = env.step(action)
            q_table[state,action] = q_table[state,action] + alpha *(reward + gamma * np.max(q_table[next_state])-q_table[state,action])
            if reward == -10:
                penalty += 1
            state = next_state
     
      
            epoch += 1
        y[i] = epoch 
