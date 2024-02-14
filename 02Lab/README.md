---
title: "Lab2: Simulation With a Pre-Trained Agent in MATLAB"
author: "Abhishek M J - CS21B2018"
date: "31-01-2024"
# geometry: "left=1.5cm,right=1.5cm,top=1.5cm,bottom=1.5cm"
geometry: "margin=2.5cm"
---


# Load the environment and agent
```matlab
rng(123)              % Set random seed for reproducibility
load robotmodel.mat   % Load the robot model
agent                 % Display the agent
env                   % Display the environment
```

![](img1.png)

# Simulate the environment
```matlab
simout = sim(agent, env)   % Simulate the environment
```

![](img2.png)

# Visualize the simulation results
```matlab
obs = simout.Observation.obs1.Data   % Get the observation data
obsmat = squeeze(obs)                % Convert the observation data to a matrix
x = obsmat(1,:);                     % Extract the x-coordinate
y = obsmat(2,:);                     % Extract the y-coordinate
plot(x,y)                            % Plot the trajectory

imout.Observation.obs1.Time;    % Get the time data
plot(t,obsmat(6,:))                  % Plot the observation data against time
```

![](img3.png)
![](img4.png)

# Visualize the action
```matlab
act = squeeze(simout.Action.act1.Data);    % Get the action data
t = simout.Action.act1.Time;               % Get the time data
Ftrans = act(1,:);                         % Extract the translational force
Frot = act(2,:);                           % Extract the rotational force
plot(t, Ftrans, t, Frot)                   % Plot the action data against time
```

![](img5.png)

# Visualize the reward
```matlab
% Change the SimulationOptions to get the reward data
opts = rlSimulationOptions("MaxSteps", 100, "NumSimulations", 5);

simout = sim(agent, env, opts)         % Simulate the environment
s = simout(1)                          % Get the first simulation
t = s.Reward.Time                      % Get the time data
r = s.Reward.Data                      % Get the reward data
plot(t,r)                              % Plot the reward data against time

% Plot the reward data for all simulations
figure
hold on
for k = 1:5
    s = simout(k);
    t = s.Reward.Time;
    r = s.Reward.Data;
    plot(t,r)
end
hold off
```
![](img6.png)