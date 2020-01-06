PROBLEM:
 We aim to create a self-driving cab, its job is to pick up the passenger at one location and drop them off in another. This cab will also take care of: Drop off the passenger to the right location. Save passenger's time by taking minimum time possible to drop off Take care of passenger's safety and traffic rules.

ENVIROMENT:

![enviroment](/image/enviroment.png)

ALGORITHM:
~~~
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
~~~

~~~
def q_learning_lambda(episode_limit,q_table,alpha =0.1,gamma = 0.6,lam = 0.5):
    y = np.zeros((episode_limit,1))

    for i in range(1,episode_limit):
        state = env.reset() #every single time we reset the enviroment
        epoch,penalty,reward =0,0,0
        done = False
        e = np.zeros([env.observation_space.n,env.action_space.n])
   
        while not done:
            action = choose_a(state)
            next_state,reward,done,info = env.step(action)
            next_action = choose_a(next_state)
            best_a = np.argmax(q_table[next_state])
            error = reward + gamma * q_table[next_state,best_a] - q_table[state,action]
            e[state,action] += 1
            q_table[state,action] += alpha * error * e[state,action]
            if next_action == best_a:
                e[state,action] = gamma * lam * e[state,action]
            else:
                e[state,action] = 0
            if reward == -10:
                penalty += 1
            state = next_state
     
      
            epoch += 1
        y[i] = epoch
~~~

RESULT:
Through learning from the past, the cab now can pick up and drop off passenger in less than 20 steps.

![result](/image/q_learning.png)

![result](/image/q_learning_lambda.png)

