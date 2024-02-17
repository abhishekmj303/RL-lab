---
title: "Lab3: Defining Environment in MATLAB"
author: "Abhishek M J - CS21B2018"
date: "14-02-2024"
fontsize: "14pt"
# geometry: "left=1.5cm,right=1.5cm,top=1.5cm,bottom=1.5cm"
geometry: "margin=2.5cm"
---

# Define Observation Vector
```matlab
obsDim = [6, 1];                    % Define the observation dimension
obsInfo = rlNumericSpec(obsDim);    % Create the observation info
```

## obsDim

- The first element, 6, specifies the number of features or dimensions in the observation.
- The second element, 1, indicates that the observation is a single data point (not a sequence of data points).

## obsInfo

- The rlNumericSpec object is designed for use with RL agents in MATLAB.
- By passing obsDim as input, it specifies the dimension

![](img1.png)

# Define Action Vector
```matlab
actDim = [2, 1];                    % Define the action dimension
actInfo = rlNumericSpec(actDim, "LowerLimit", -1, "UpperLimit", 1);    % Create the action info
```

## actDim

- The first element, 2, specifies the number of features or dimensions in the action.
- The second element, 1, indicates that the action is a single data point (not a sequence of data points).

## actInfo

- The rlNumericSpec object is designed for use with RL agents in MATLAB.
- By passing actDim as input, it specifies the dimension
- The "LowerLimit" and "UpperLimit" name-value pairs specify the lower and upper limits of the action space, respectively.

![](img2.png)

# Define Simulation Environment
```matlab
env = rlSimulinkEnv("whrobot", "whrobot/controller", obsInfo, actInfo);    % Create the environment
env.ResetFcn = @randomstart;    % Set the reset function
```

## env

- The rlSimulinkEnv creates a simulation environment object (env) using the rlSimulinkEnv function, specifically designed for integrating Simulink models with RL agents in MATLAB.
- By passing "whrobot" and "whrobot/controller" as input, it specifies the Simulink model and the controller block within the model.
- The obsInfo and actInfo objects specify the observation and action information, respectively.
- The ResetFcn property specifies the reset function for the environment.
- The @randomstart function handle specifies the reset function.

![](img3.png)

# Load and Simulate the Environment
```matlab
rng(123)              % Set random seed for reproducibility
load robotmodel agent    % Load the robot model and agent
sim(agent, env)         % Simulate the environment
```

- The rng function sets the random seed to 123 for reproducibility.
- The load function loads the robot model and agent from the robotmodel.mat file.
- The sim function simulates the environment using the loaded agent and environment.

![](img4.png)

![](img5.png)