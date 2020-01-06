# self-driven-cab
PROBLEM:
We aim to create a self-driving cab, its job is to pick up the passenger at one location and drop them off in another. This cab will also take care of:
Drop off the passenger to the right location.
Save passenger's time by taking minimum time possible to drop off
Take care of passenger's safety and traffic rules

The main idea of this project is to learning from the past and solve the problemfaster
ALGORITHMS:
#brute-force our way to solving the problem without RL
env.s = 242
epochs = 0 #count the total number of steps
penalties, reward = 0,0
frames = [] #for animation
done = False #[1.0,342,-1,False]
while not done:
    action = env.action_space.sample()
    state,reward,done,info = env.step(action)
    if reward == -10:
        penalties +=1
    frames.append({
        'frame': env.render(mode='ansi'),
        'state':state,
        'action':action,
        'reward':reward
         }
     )
    epochs +=1
print("Timesteps taken: {}".format(epochs))
print("Penalty happens: {}".format(penalties))
#this is because we arent learning from past experience.

#gonna write a function here
#Parameters
#alpha = 0.1
#gamma = 0.6
epsilon = 0.1
#greedy policy
q_table = np.zeros([env.observation_space.n,env.action_space.n])
#gonna write a function here 
episode_limit = 10001
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
