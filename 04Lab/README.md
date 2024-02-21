---
title: "Lab 4: Creating Simple MDP MATLAB Environment with a Q-Learning Agent"
author: "Abhishek M J - CS21B2018"
date: "21-02-2024"
fontsize: "14pt"
# geometry: "left=1.5cm,right=1.5cm,top=1.5cm,bottom=1.5cm"
geometry: "margin=2.5cm"
---

# MDP Matlab Environment

```matlab
MDP = createMDP(8, ["left"; "right"])
```
- This line creates a Markov Decision Process (MDP) object using the createMDP function.
- The first argument, 8, specifies the number of states in the MDP. Imagine eight distinct positions or situations in your environment.
- The second argument, ["left"; "right"], defines the possible actions that can be taken in each state. In this case, you can either move "left" or "right".

![](img01.png)

```matlab
MDP.States
```
- It displays the numbers from 1 to 8, representing the unique identifiers for each state.

![](img02.png)

```matlab
MDP.Actions
```
- It shows "left" and "right", the two actions you can take from any state.

![](img03.png)

```matlab
MDP.T
```
- Each element in the array MDP.T(s, a, s') tells you the probability of transitioning from state s to state s' when action a is taken.

![](img04.png)

```matlab
MDP.R
```
- Each element in the array MDP.R(s, a) tells you the reward you receive immediately after taking action a in state s.

![](img05.png)


# Defining Rewards and Transitions Probabilities

```matlab
MDP.TerminalStates = ["s1";"s8"];
```
- This line specifies that states 1 and 8 are terminal states. Once the agent reaches one of these states, the episode ends.

```matlab
nS = numel(MDP.States);
nA = numel(MDP.Actions);
```
- This line calculates the number of states and actions in the MDP.

```matlab
MDP.R = -1*ones(nS,nA);
```
- This line initializes the reward array with -1 for all state-action pairs.

```matlab
MDP.R(:, state2idx(MDP, MDP.TerminalStates), :) = 10;
```
- This line sets the reward to 10 for all state-action pairs that lead to a terminal state.

```matlab
MDP.T(1, 1, 1) = 1;
MDP.T(1, 2, 2) = 1;

MDP.T(2, 1, 1) = 1;
MDP.T(2, 3, 2) = 1;

MDP.T(3, 2, 1) = 1;
MDP.T(3, 4, 2) = 1;

MDP.T(4, 3, 1) = 1;
MDP.T(4, 5, 2) = 1;

MDP.T(5, 4, 1) = 1;
MDP.T(5, 6, 2) = 1;

MDP.T(6, 5, 1) = 1;
MDP.T(6, 7, 2) = 1;

MDP.T(7, 6, 1) = 1;
MDP.T(7, 8, 2) = 1;

MDP.T(8, 7, 1) = 1;
MDP.T(8, 8, 2) = 1;
```
- This block of code sets the transition probabilities for each state-action pair. For example, MDP.T(1, 2, 2) = 1 sets the probability of transitioning from state 1 to state 2 when action 2 ("right") is taken to 1.
- In simpler terms, the agent moves to right from state 1 it will always (with probability 1) reach state 2.

```matlab
MDP.T
MDP.R
```

```matlab
env = rlMDPEnv(MDP)
```
- This line creates a reinforcement learning environment using the rlMDPEnv function. The environment is based on the MDP object we created earlier.


# Define Q-Table and Initialize Agent

```matlab
state_information = getObservationInfo(env)
action_information = getActionInfo(env)
```
- This line retrieves the observation and action information from the environment.

![](img06.png)

```matlab
qTable = rlTable(state_information, action_information)
qTable.Table
```
- This line creates a Q-table using the rlTable function. The Q-table is a matrix that stores the Q-values for each state-action pair.

![](img07.png)

```matlab
qTable.Table = ones(size(qTable.Table))*5
qTable.Table
```
- This line initializes the Q-table with a constant value of 5.

![](img08.png)

```matlab
qRepresentation = rlQValueRepresentation(qTable, state_information, action_information)
qRepresentation.Options
qRepresentation.Options.L2RegularizationFactor = 0;
qRepresentation.Options.LearnRate = 0.01;
```
- This block of code creates a Q-value representation using the rlQValueRepresentation function. The Q-value representation is used to define the learning parameters for the Q-learning agent.
- The L2RegularizationFactor and LearnRate options are set to 0 and 0.01, respectively.

![](img09.png)

```matlab
agentOpts = rlQAgentOptions
agentOpts.EpsilonGreedyExploration
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.01
qAgent = rlQAgent(qRepresentation, agentOpts)
```
- This block of code creates a Q-learning agent using the rlQAgent function. The agent uses the Q-value representation and agent options we defined earlier.
- The EpsilonGreedyExploration.EpsilonDecay option is set to 0.01, which means the exploration rate decreases by 0.01 after each episode.
- The exploration rate determines the probability of the agent taking a random action instead of the action with the highest Q-value.

![](img10.png)

# Train the Q-Learning Agent

```matlab
trainOpts = rlTrainingOptions
trainOpts.MaxStepsPerEpisode = 10;
trainOpts.MaxEpisodes = 100;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 13;
trainOpts.ScoreAveragingWindowLength = 30;
```
- This block of code creates training options using the rlTrainingOptions function. The training options specify the maximum number of steps and episodes, as well as the stopping criteria for training.

![](img11.png)

```matlab
QTable0 = getLearnableParameters(getCritic(qAgent));
disp(QTable0{1})
```
- This line retrieves the initial Q-table from the Q-learning agent.

![](img12.png)

```matlab
doTraining = true;
if doTraining
    trainingStats = train(qAgent, env, trainOpts);
else
    load('genericMDPQAgent.mat', 'qAgent')
end
```
- This block of code trains the Q-learning agent using the train function. The training process is controlled by the training options we defined earlier.
- The trainingStats variable stores the training statistics, such as the average reward per episode and the total number of steps taken.

![](img13.png)

```matlab
Data = sim(qAgent, env)
cumulativeReward = sum(Data.Reward)
```
- This line simulates the Q-learning agent in the environment using the sim function. The Data variable stores the observations, actions, and rewards collected during the simulation.
- The cumulativeReward variable calculates the total reward obtained by the agent during the simulation.

```matlab
QTable1 = getLearnableParameters(getCritic(qAgent));
disp(QTable1{1})
```
- This line retrieves the final Q-table from the Q-learning agent after training.

![](img14.png)