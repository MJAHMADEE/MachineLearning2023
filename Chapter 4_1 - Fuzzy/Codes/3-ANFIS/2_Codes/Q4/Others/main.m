%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ102
% Project Title: Time-Series Prediction using ANFIS
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Create Time-Series Data

% Global Ice Volume over the last 440,000 years
data = load('global_ice_volume');
x = data.x;

Delays = [1 2 3 4 5];
[Inputs, Targets] = CreateTimeSeriesData(x, Delays);
Inputs = Inputs';
Targets = Targets';

nData = size(Inputs,1);

%% Shuffling Data

PERM = 1:nData; % Permutation to Shuffle Data

pTrain=0.75;
nTrainData=round(pTrain*nData);
TrainInd=PERM(1:nTrainData);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);

pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInd=PERM(nTrainData+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);

%% Selection of FIS Generation Method

Option{1}='Grid Partitioning (genfis1)';
Option{2}='Subtractive Clustering (genfis2)';
Option{3}='FCM (genfis3)';

ANSWER=questdlg('Select FIS Generation Approach:',...
                'Select GENFIS',...
                Option{1},Option{2},Option{3},...
                Option{3});
pause(0.01);

%% Setting the Parameters of FIS Generation Methods

switch ANSWER
    case Option{1}
        Prompt={'Number of MFs','Input MF Type:','Output MF Type:'};
        Title='Enter genfis1 parameters';
        DefaultValues={'2', 'gaussmf', 'linear'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.01);

        nMFs=str2num(PARAMS{1});	%#ok
        InputMF=PARAMS{2};
        OutputMF=PARAMS{3};
        
        fis=genfis1([TrainInputs TrainTargets],nMFs,InputMF,OutputMF);

    case Option{2}
        Prompt={'Influence Radius:'};
        Title='Enter genfis2 parameters';
        DefaultValues={'0.55'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.01);

        Radius=str2num(PARAMS{1});	%#ok
        
        fis=genfis2(TrainInputs,TrainTargets,Radius);
        
    case Option{3}
        Prompt={'Number fo Clusters:',...
                'Partition Matrix Exponent:',...
                'Maximum Number of Iterations:',...
                'Minimum Improvemnet:'};
        Title='Enter genfis3 parameters';
        DefaultValues={'10', '2', '100', '1e-5'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.01);

        nCluster=str2num(PARAMS{1});        %#ok
        Exponent=str2num(PARAMS{2});        %#ok
        MaxIt=str2num(PARAMS{3});           %#ok
        MinImprovment=str2num(PARAMS{4});	%#ok
        DisplayInfo=1;
        FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
        
        fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);
end

%% Training ANFIS Structure

Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'100', '0', '0.01', '0.9', '1.1'};

PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
pause(0.01);

MaxEpoch=str2num(PARAMS{1});                %#ok
ErrorGoal=str2num(PARAMS{2});               %#ok
InitialStepSize=str2num(PARAMS{3});         %#ok
StepSizeDecreaseRate=str2num(PARAMS{4});    %#ok
StepSizeIncreaseRate=str2num(PARAMS{5});    %#ok
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);


%% Apply ANFIS to Data

Outputs=evalfis(Inputs,fis);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);

%% Error Calculation

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors.^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors.^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');

figure;
PlotResults(Targets,Outputs,'All Data');

if ~isempty(which('plotregression'))
    figure;
    plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
                   TestTargets, TestOutputs, 'Test Data', ...
                   Targets, Outputs, 'All Data');
    set(gcf,'Toolbar','figure');
end
