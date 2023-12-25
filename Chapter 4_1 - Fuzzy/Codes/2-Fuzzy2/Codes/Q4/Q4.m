clc;
clear;
close all
%% Generating Data
% First Dataset
dataset_1 = [0      1       0       -19.00;...
             1      1.95    9.37    -17.95;...
             2      2.88    18.23   -16.90;...
             3      3.79    26.57   -15.85;...
             4      4.65    34.44   -14.80;...
             5      5.45    41.78   -13.75;...
             6      6.18    48.60   -12.70;...
             7      7.48    54.91   -11.65;...
             8      7.99    60.71   -10.60;...
             9      8.72    65.99   -9.55;...
             10     9.01    70.75   -8.50;...
             11     9.28    74.98   -7.45;...
             12     9.46    78.70   -6.40;...
             13     9.59    81.90   -5.34;...
             14     9.72    84.57   -4.30;...
             15     9.81    86.72   -3.25;...
             16     9.88    88.34   -2.20;...
             17     9.91    89.44   0];

% Second Dataset
dataset_2 = [0      1       90       11.5;...
             1      1.10    79.08    9.40;...
             2      1.29    74.40    8.35;...
             3      1.56    70.24    7.30;...
             4      1.90    66.60    6.25;...
             5      2.29    63.48    5.20;...
             6      2.73    60.88    4.15;...
             7      3.22    58.81    3.10;...
             8      3.74    57.26    2.05;...
             9      4.84    55.74   -0.05;...
             10     5.96    56.31   -2.15;...
             11     6.51    57.38   -3.20;...
             12     7.05    58.98   -4.25;...
             13     7.56    61.10   -5.30;...
             14     8.04    63.75   -6.35;...
             15     8.48    66.92   -7.40;...
             16     8.87    70.61   -8.45;...
             17     9.20    74.82   -7.40;...
             18     9.46    78.51   -6.35;...
             19     9.66    81.68   -5.30;...
             20     9.80    84.33   -4.25;...
             21     9.90    86.45   -3.20;...
             22     9.96    88.05   -2.15;...
             23     10.01   89.67    0];

[n1,~] = size(dataset_1);
[n2,~] = size(dataset_2);

n = n1 + n2;

dataset = zeros(n,5);
dataset(1:n1,2:4) = dataset_1(:,2:4);
dataset(n1+1:n,2:4) = dataset_2(:,2:4);

%% Asssign Degree to Each Rule
% Five Membership Functions for Iput x
N_Rule_x = 5;
S2_x = [0 0 1.5 7];
S1_x = [4 7 10];
CE_x = [9 10 11];
B1_x = [10 13 16];
B2_x = [13 18.5 20 20];

% Seven Membership Functions for Iput phi
N_Rule_phi = 7;
S3_phi = [-115 -65 -15];
S2_phi = [-45 0 45];
S1_phi = [15 52.5 90];
CE_phi = [80 90 100];
B1_phi = [90 127.5 165];
B2_phi = [135 180 225];
B3_phi = [180 225 295];

% Seven Membership Functions for Iput theta
N_Rule_theta = 7;
S3_theta = [-60 -40 -20];
S2_theta = [-33 -20 -7];
S1_theta = [-14 -7 0];
CE_theta = [-4 0 4];
B1_theta = [0 7 14];
B2_theta = [7 20 33];
B3_theta = [20 40 60];

%% Prealloctions

Rules = zeros(n,4);
Rules_Total = zeros(n,4);
vec_x = zeros(1,N_Rule_x);
vec_phi = zeros(1,N_Rule_phi);
vec_theta = zeros(1,N_Rule_theta);

for k=1:n
    dataset(k,1) = k;
    x_k = dataset(k,2);
    vec_x = [trapmf(x_k,S2_x) trimf(x_k,S1_x) trimf(x_k,CE_x) trimf(x_k,B1_x) trapmf(x_k,B2_x)];
    phi_k = dataset(k,3);
    vec_phi = [trimf(phi_k,S3_phi) trimf(phi_k,S2_phi) trimf(phi_k,S1_phi) trimf(phi_k,CE_phi) trimf(phi_k,B1_phi) trimf(phi_k,B2_phi) trimf(phi_k,B3_phi)];
    theta_k = dataset(k,4);
    vec_theta = [trimf(theta_k,S3_theta) trimf(theta_k,S2_theta) trimf(theta_k,S1_theta) trimf(theta_k,CE_theta) trimf(theta_k,B1_theta) trimf(theta_k,B2_theta) trimf(theta_k,B3_theta)];
    [value_x,column_x] = max(vec_x);
    [value_phi,column_phi] = max(vec_phi);
    [value_theta,column_theta] = max(vec_theta);

    vec = [max(vec_x) max(vec_phi) max(vec_theta)];
    dataset(k,5) = prod(vec);

    Rules(k,1:4) = [column_x column_phi column_theta prod(vec)];
end

%% Delete Extra Rules

Rules_Total(1,1:4) = Rules(1,1:4);
i=1;
for t=2:n
    m = zeros(1,i);
    for j=1:i
        m(1,j) = isequal(Rules(t,1:2),Rules_Total(j,1:2));
            if m(1,j)==1 && Rules(t,4)>=Rules_Total(j,4)
                Rules_Total(j,1:4) = Rules(t,1:4);
            end
    end
            if sum(m) == 0
                Rules_Total(i+1,1:4) = Rules(t,1:4);
                i = i+1;
            end
end

Final_Rules = Rules_Total(1:i,:)

%% Create Fuzzy Inference System

Fisname = 'Controller';
Fistype = 'mamdani';
Andmethod='prod';
Ormethod='max';
Impmethod='prod';
Aggmethod='max';
Defuzzmethod='centroid';

fis = newfis(Fisname,Fistype,Andmethod,Ormethod,Impmethod,Aggmethod,Defuzzmethod);

%% Add Variables

fis = addvar(fis,'input','x',[0 20]);
fis = addvar(fis,'input','phi',[-90 270]);
fis = addvar(fis,'output','theta',[-40 40]);

%% Add Membership Function

fis = addmf(fis,'input',1,'S2','trapmf',S2_x);
fis = addmf(fis,'input',1,'S1','trimf',S1_x);
fis = addmf(fis,'input',1,'CE','trimf',CE_x);
fis = addmf(fis,'input',1,'B1','trimf',B1_x);
fis = addmf(fis,'input',1,'B2','trapmf',B2_x);

fis = addmf(fis,'input',2,'S3','trimf',S3_phi);
fis = addmf(fis,'input',2,'S2','trimf',S2_phi);
fis = addmf(fis,'input',2,'S1','trimf',S1_phi);
fis = addmf(fis,'input',2,'CE','trimf',CE_phi);
fis = addmf(fis,'input',2,'B1','trimf',B1_phi);
fis = addmf(fis,'input',2,'B2','trimf',B2_phi);
fis = addmf(fis,'input',2,'B3','trimf',B3_phi);

fis = addmf(fis,'output',1,'S3','trimf',S3_theta);
fis = addmf(fis,'output',1,'S2','trimf',S2_theta);
fis = addmf(fis,'output',1,'S1','trimf',S1_theta);
fis = addmf(fis,'output',1,'CE','trimf',CE_theta);
fis = addmf(fis,'output',1,'B1','trimf',B1_theta);
fis = addmf(fis,'output',1,'B2','trimf',B2_theta);
fis = addmf(fis,'output',1,'B3','trimf',B3_theta);

%% Add Rules
fis_Rules = ones(i,5);
fis_Rules(1:i,1:3) = Rules_Total(1:i,1:3);
fis = addrule(fis,fis_Rules);

%% Plot & Analysis

% Plot Membership Function
figure(1)
plotmf(fis,'input',1)
figure(2)
plotmf(fis,'input',2)
figure(3)
plotmf(fis,'output',1)
figure(4)
gensurf(fis)

% Controlling the Truck from an Arbitrary Initial Point (x0,phi0)

% Truck Parameter
b = 4;
Data_Table = zeros(n,4);

x = zeros(1,n);
phi = zeros(1,n);
y = zeros(1,n);
y(1,1) = 2;

x(1,1) = input('Enter Initial Point for x(0<x<20) & Press Enter Key: ');
phi(1,1) = input('Enter Initial Point for phi(-90<phi<270 degree) and then Press Enter Key: ');

for t=1:n
    theta = evalfis([x(1,t);phi(1,t)],fis);
    Data_Table(t,:) = [t,x(1,t),phi(1,t),theta];
    x(1,t+1) = x(1,t)+cos((phi(1,t)+theta).*pi/180)+sin(theta*pi/180)*sin(phi(1,t).*pi/180); ...
    phi(1,t+1) = phi(1,t) - (asin(2*sin(theta*pi/180)/b))*180/pi;
    y(1,t+1) = y(1,t)+sin((phi(1,t)+theta)*pi/180)-sin(theta*pi/180)*cos(phi(1,t)*pi/180);
end

disp('_____________________________')
disp(['x_Final = ',num2str(x(end))])
disp(['phi_Final = ',num2str(phi(end))])
disp('_____________________________')

figure(5) = figure('Color', [1 1 1]);
plot(x,y,'Linewidth',2)
xlabel('x')
ylabel('y');
axis([0 20 -10 50])
grid on