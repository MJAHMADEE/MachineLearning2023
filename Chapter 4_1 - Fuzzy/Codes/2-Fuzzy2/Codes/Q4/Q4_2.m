clc;
clear;
close all

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

fis = addmf(fis,'input',1,'S2','trapmf',[0 0 1.5 7]);
fis = addmf(fis,'input',1,'S1','trimf',[4 7 10]);
fis = addmf(fis,'input',1,'CE','trimf',[9 10 11]);
fis = addmf(fis,'input',1,'B1','trimf',[10 13 16]);
fis = addmf(fis,'input',1,'B2','trapmf',[13 18.5 20 20]);

fis = addmf(fis,'input',2,'S3','trimf',[-115 -65 -15]);
fis = addmf(fis,'input',2,'S2','trimf',[-45 0 45]);
fis = addmf(fis,'input',2,'S1','trimf',[15 52.5 90]);
fis = addmf(fis,'input',2,'CE','trimf',[80 90 100]);
fis = addmf(fis,'input',2,'B1','trimf',[90 127.5 165]);
fis = addmf(fis,'input',2,'B2','trimf',[135 180 225]);
fis = addmf(fis,'input',2,'B3','trimf',[180 225 295]);

fis = addmf(fis,'output',1,'S3','trimf',[-60 -40 -20]);
fis = addmf(fis,'output',1,'S2','trimf',[-33 -20 -7]);
fis = addmf(fis,'output',1,'S1','trimf',[-14 -7 0]);
fis = addmf(fis,'output',1,'CE','trimf',[-4 0 4]);
fis = addmf(fis,'output',1,'B1','trimf',[0 7 14]);
fis = addmf(fis,'output',1,'B2','trimf',[7 20 33]);
fis = addmf(fis,'output',1,'B3','trimf',[20 40 60]);

%% Add Rules
Rules = [1 1 2 1 1;...
         1 2 2 1 1;...
         1 3 5 1 1;...
         1 4 6 1 1;...
         1 5 6 1 1;...
         2 1 1 1 1;...
         2 2 1 1 1;...
         2 3 3 1 1;...
         2 4 6 1 1;...
         2 5 7 1 1;...
         2 6 7 1 1;...
         3 2 1 1 1;...
         3 3 2 1 1;...
         3 4 4 1 1;...
         3 5 6 1 1;...
         3 6 7 1 1;...
         4 2 1 1 1;...
         4 3 1 1 1;...
         4 4 2 1 1;...
         4 5 5 1 1;...
         4 6 7 1 1;...
         4 7 7 1 1;...
         5 3 2 1 1;...
         5 4 2 1 1;...
         5 5 3 1 1;...
         5 6 6 1 1;...
         5 7 6 1 1]
fis = addrule(fis,Rules);

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
n = 250;
Data_Table = zeros(n,5);

x = zeros(1,n);
phi = zeros(1,n);
y = zeros(1,n);
y(1,1) = 2;

x(1,1) = input('Enter Initial Point for x(0<x<20) & Press Enter Key: ');
phi(1,1) = input('Enter Initial Point for phi(-90<phi<270 degree) and then Press Enter Key: ');

x_desired = 10;
phi_desired = 90;

% Cost Function
J = norm([x_desired-x(1,1) phi_desired-phi(1,1)])

t=1;
while(J>=0.01)
    theta = evalfis([x(1,t);phi(1,t)],fis);
    Data_Table(t,:) = [t-1,x(1,t),y(1,t),phi(1,t),theta];
    x(1,t+1) = x(1,t)+cos((phi(1,t)+theta).*pi/180)+sin(theta*pi/180)*sin(phi(1,t).*pi/180); ...
    phi(1,t+1) = phi(1,t) - (asin(2*sin(theta*pi/180)/b))*180/pi;
    y(1,t+1) = y(1,t)+sin((phi(1,t)+theta)*pi/180)-sin(theta*pi/180)*cos(phi(1,t)*pi/180);

    J = norm([x_desired-x(1,t+1) ...
        phi_desired - phi(1,t+1)]);
    t = t+1
end

x_truck(1,:) = x(1,1:t);
phi_truck(1,:) = phi(1,1:t);
y_truck(1,:) = y(1,1:t);

disp('_____________________________')
disp(['x_Final = ',num2str(x_truck(end))])
disp(['phi_Final = ',num2str(phi_truck(end))])
disp(['y_Final = ',num2str(y_truck(end))])
disp('_____________________________')

figure(5) = figure('Color', [1 1 1]);
plot(x_truck,y_truck,'Linewidth',2)
xlabel('x')
ylabel('y');
axis([0 20 -10 50])
grid on