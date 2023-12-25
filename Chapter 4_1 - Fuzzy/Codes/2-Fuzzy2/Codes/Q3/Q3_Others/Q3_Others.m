clc
clear all
close all

xmax=3;
xmin=-3;
x=xmin:0.01:xmax;
%Create Date
tol=xmax-xmin+1;
xi=linspace(xmin,xmax,tol);
for i=1:tol
 data(i)=sin(xi(i));
end
%Make MFC
h=1;
a=xmin-h;
b=xmin;
c=xmin+h;
for i=1:tol
 mu(i,:)=trimf(x,[a b c]);
 a=a+h;
 b=b+h;
 c=c+h;
end
% Plot MFC
for i=1:tol
 plot(x,mu(i,:))
 hold on
end
num=0;
den=0;
for i=1:tol
 num=num+data(i)*mu(i,:);
 den=den+mu(i);
end
fx=num./den;
figure,plot(x,fx)
hold on
plot(x,sin(x),'r--','LineWidth',2)
legend('f(x)','g(x)')
