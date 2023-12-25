clc;
clear;
close all;

%% Load Data

f=@(x,y) sin(x)./x .* sin(y)./y;

xmin=-10;
xmax=10;
ymin=-10;
ymax=10;
x=linspace(xmin,xmax,26)';
y=linspace(ymin,ymax,26)';
F=f(x,y);

TrainInputs=[x,y];
TrainTargets=F;
TrainData=[TrainInputs TrainTargets];

xx=linspace(xmin,xmax,106)';
yy=linspace(ymin,ymax,106)';
FF=f(xx,yy);

TestInputs=[xx,yy];
TestTargets=FF;
TestData=[TestInputs TestTargets];


%% Design ANFIS

% nMFs=5;
% InputMF='gaussmf';
% OutputMF='linear';
% 
% fis=genfis1(TrainData,nMFs,InputMF,OutputMF);

fis=genfis2(TrainInputs,TrainTargets,0.2);

MaxEpoch=100;
ErrorGoal=0;
InitialStepSize=0.01;
StepSizeDecreaseRate=0.9;
StepSizeIncreaseRate=1.1;
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
            
fis=anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);


%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

%% Apply ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');