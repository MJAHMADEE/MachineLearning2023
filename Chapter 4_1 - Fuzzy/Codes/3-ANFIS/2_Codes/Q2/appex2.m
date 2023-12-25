clc;
clear;
close all;

%% Load Data

f=@(x,y,z) (1+x.^0.5+y.^-1+z.^(-1.5)).^2;

xmin=1;
xmax=6;
ymin=1;
ymax=6;
zmin=1;
zmax=6;
x=linspace(xmin,xmax,26)';
y=linspace(ymin,ymax,26)';
z=linspace(zmin,zmax,26)';
F=f(x,y,z);

TrainInputs=[x,y,z];
TrainTargets=F;
TrainData=[TrainInputs TrainTargets];

xx=linspace(xmin,xmax,101)';
yy=linspace(ymin,ymax,101)';
zz=linspace(zmin,zmax,101)';
FF=f(xx,yy,zz);

TestInputs=[xx,yy,zz];
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
TrainAPE= abs(TrainErrors(:))./abs(TrainTargets(:))*(26/100)

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

%% Apply ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);
TestAPE= abs(TestErrors(:))./abs(TestTargets(:))*(101/100)

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');
