MDP = createMDP(8, ["left"; "right"])

MDP.States

MDP.Actions

MDP.T

MDP.R


MDP.TerminalStates = ["s1";"s8"];

nS = numel(MDP.States);
nA = numel(MDP.Actions);

MDP.R = -1*ones(nS,nS,nA);
MDP.R(:, state2idx(MDP, MDP.TerminalStates), :) = 10;

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

MDP.T

MDP.R

env = rlMDPEnv(MDP);


state_information = getObservationInfo(env)

action_information = getActionInfo(env)

qTable = rlTable(state_information, action_information)

qTable.Table

qTable.Table = ones(size(qTable.Table))*5

qTable.Table

qRepresentation = rlQValueRepresentation(qTable, state_information, action_information)

qRepresentation.Options

qRepresentation.Options.L2RegularizationFactor = 0;
qRepresentation.Options.LearnRate = 0.01;

agentOpts = rlQAgentOptions

agentOpts.EpsilonGreedyExploration

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.01

qAgent = rlQAgent(qRepresentation, agentOpts)

trainOpts = rlTrainingOptions

trainOpts.MaxStepsPerEpisode = 10;
trainOpts.MaxEpisodes = 100;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 13;
trainOpts.ScoreAveragingWindowLength = 30;

QTable0 = getLearnableParameters(getCritic(qAgent));
disp(QTable0{1})


doTraining = true;

if doTraining
    trainingStats = train(qAgent, env, trainOpts);
else
    load('genericMDPQAgent.mat', 'qAgent')
end

Data = sim(qAgent, env)

cumulativeReward = sum(Data.Reward)

QTable = getLearnableParameters(getCritic(qAgent));
QTable{1}
