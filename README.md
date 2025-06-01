# VALUE ITERATION ALGORITHM

## AIM

To implement a Python program using the Value Iteration algorithm to determine the optimal policy for an agent navigating the Frozen Lake environment in OpenAI Gym.

## PROBLEM STATEMENT

The Frozen Lake problem is a classic reinforcement learning task where an agent must learn to navigate a slippery gridworld to reach a goal state without falling into holes. The environment is represented as a 4x4 grid where:

- S is the starting state,

- F is a frozen safe tile,

- H is a hole (falling ends the episode), and

- G is the goal state.

The agent can take one of four actions at each state: Left, Down, Right, or Up. However, due to the slippery nature of the environment, the agent may not always move in the intended direction. The goal is to compute the optimal policy that maximizes the expected return using the Value Iteration algorithm.


## VALUE ITERATION ALGORITHM

![image](https://github.com/user-attachments/assets/f9503116-00e7-4b7b-b60c-0316e8e4f372)


## VALUE ITERATION FUNCTION
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def value_iteration(P, gamma=0.99, theta=1e-10):
    n_states = len(P)
    n_actions = len(P[0])
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            A = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    A[a] += prob * (reward + gamma * V[next_state])
            max_val = np.max(A)
            delta = max(delta, np.abs(V[s] - max_val))
            V[s] = max_val
        if delta < theta:
            break

    # Extract optimal policy
    def pi_opt(s):
        A = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in P[s][a]:
                A[a] += prob * (reward + gamma * V[next_state])
        return np.argmax(A)

    return V, pi_opt
```

## OUTPUT:

### Mention the optimal policy, optimal value function , success rate for the optimal policy.


![image](https://github.com/user-attachments/assets/f2b5631c-307e-43d9-aa4a-0d38e6d590b4)


![image](https://github.com/user-attachments/assets/0aa7955f-7b54-416b-93d9-b57b5d17473f)


![image](https://github.com/user-attachments/assets/0c11a429-2e4a-4f13-acff-ac7966b8c1fe)


## RESULT:

Thus , to implement a Python program using the Value Iteration algorithm to determine the optimal policy for an agent navigating the Frozen Lake environment in OpenAI Gym is successfully implemented.

