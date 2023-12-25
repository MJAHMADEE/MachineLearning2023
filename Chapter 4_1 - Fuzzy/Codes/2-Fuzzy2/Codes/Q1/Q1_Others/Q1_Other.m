clc
clear all
close all

syms X1 X2
G=1/(1+X1^2+X2^2);
d1=diff(G,X1);pretty(d1)
d11=diff(G,X1,2);pretty(d11)
d2=diff(G,X2);pretty(d2)
d22=diff(G,X2,2);pretty(d22)
x1=linspace(-1,1,201);
x2=linspace(-1,1,201);
x=linspace(-1,1,201);
g=1./(1+x1.^2+x2.^2);
x_min=-1;
x_max=1;
%% First Order
% Create Date
xi1_star=linspace(-1,1,41);
xi2_star=linspace(-1,1,41);
for i1=1:41
 for i2=1:41
 data(i1,i2)=1/(1+xi1_star(i1)^2+xi2_star(i2)^2);
 end
end
% Input MemberShip Function
h=0.05;
a=x_min-h;
b=x_min;
c=x_min+h;
for row=1:41
u_a1(row,:)=trimf(x,[a b c]);
 u_a2(row,:)=trimf(x,[a b c]);
 a=a+h;
 b=b+h;
 c=c+h;
end
% Plotting Input MemberShip Function
figure
subplot(2,1,1)
plot(x,u_a1(1,:),'k','LineWidth',2)
hold on
title('\mu_{A_1}','FontSize',16)
subplot(2,1,2)
plot(x,u_a2(1,:),'k','LineWidth',2)
hold on
title('\mu_{A_2}','FontSize',16)
for i=2:41
 subplot(2,1,1)
 plot(x,u_a1(i,:),'k','LineWidth',2)
 subplot(2,1,2)
 plot(x,u_a2(i,:),'k','LineWidth',2)
end
subplot(2,1,1),hold off
subplot(2,1,2),hold off
num=0;
den=0;
for i1=1:41
 for i2=1:41
 num=num+data(i1,i2).*u_a1(i1,:).*u_a2(i2,:);
 den=den+u_a1(i1,:).*u_a2(i2,:);
 end
end
f1_x=num./den;
figure,plot(x,f1_x,'k','LineWidth',2)
title('f(x) --> First Order','FontSize',16)
figure,plot(x,g,'-k','Linewidth',2)
title('g(x)','FontSize',16)
%% Second Order
% Create Date
xi1_star=linspace(-1,1,9);
xi2_star=linspace(-1,1,9);
for i1=1:9
 for i2=1:9
 data(i1,i2)=1/(1+xi1_star(i1)^2+xi2_star(i2)^2);
 end
end
% Input MemberShip Function
h=0.25;
a=x_min-h;
b=x_min;
c=x_min+h;
for row=1:9
 u_a1(row,:)=trimf(x,[a b c]);
 u_a2(row,:)=trimf(x,[a b c]);
 a=a+h;
 b=b+h;
 c=c+h;
end
% Plotting Input MemberShip Function
figure(1)
subplot(2,1,1)
plot(x,u_a1(1,:),'k','LineWidth',2)
hold on
title('\mu_{A_1}','FontSize',16)
subplot(2,1,2)
plot(x,u_a2(1,:),'k','LineWidth',2)
hold on
title('\mu_{A_2}','FontSize',16)
for i=2:9
 subplot(2,1,1)
 plot(x,u_a1(i,:),'k','LineWidth',2)
 subplot(2,1,2)
 plot(x,u_a2(i,:),'k','LineWidth',2)
end
num=0;
den=0;
for i1=1:9
 for i2=1:9
 num=num+data(i1,i2).*u_a1(i1,:).*u_a2(i2,:);
 den=den+u_a1(i1,:).*u_a2(i2,:);
 end
end
f2_x=num./den;
figure(2)
plot(x,f2_x,'k','LineWidth',2)
title('f2(x) --> Second Order','FontSize',16)
figure(3)
plot(x,g,'-k','Linewidth',2)
title('g(x)','FontSize',16)
% % Compare
figure(4)
plot(x,f1_x,'.k',x,f2_x,'--b',x,g,'-r')
title('Compare','FontSize',16)
legend('First Order','Second Order','g(x)')
figure(5)
plot(x,g-f1_x,'-r',x,g-f2_x,'--b')
title('Compare','FontSize',16)
legend('First Order Error','Second Order Error')