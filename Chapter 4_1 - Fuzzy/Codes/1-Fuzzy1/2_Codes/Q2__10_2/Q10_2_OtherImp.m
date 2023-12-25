clc
clear
close all

alpha = -1;
beta = 1;
x = alpha:0.001:beta;
h = 0.01;
% Center of member fcn
X = -1:h:1;
X(2,:) = sin(pi*X)+cos(pi*X)+sin(pi*X).*cos(pi*X);
sigma = 0.001;
k=1;
for x = -1:0.001:1
    [a, b]=min( abs(X(1,:)-x*ones(1,length(X))) );
    if x<X(1,b) && b>1
       mu1=exp(-(x-X(1,b))^2/sigma);
       mu2=exp(-(x-X(1,b-1))^2/sigma);
       f(k)=(mu1*X(2,b)+mu2*X(2,b-1))/(mu1+mu2);
    elseif x > X(1,b)
       mu1=exp(-(x-X(1,b))^2/sigma);
       mu2=exp(-(x-X(1,b+1))^2/sigma);
       f(k)=(mu1*X(2,b)+mu2*X(2,b+1))/(mu1+mu2);
    elseif x==X(1,b)
        f(k)=X(2,b);
    end
     k=k+1;   
end
%Plot
g_x = sin(x.*pi)+cos(x.*pi)+sin(x.*pi).*cos(x.*pi);
figure0 = figure('Color',[1 1 1]);
x=-1:.001:1;
plot(x, sin(pi*x)+cos(pi*x)+sin(pi*x).*cos(pi*x), 'b-', x, f, 'r--','Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('$g(x),f(x)$','Interpreter','latex');
legend('$g(x)$','$f(x)$','Interpreter','latex')
grid on

figure1 = figure('Color',[1 1 1]);
x=-1:.001:1;
plot(x,f,'Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('f(x)','Interpreter','latex');
legend('$f(x)$','Interpreter','latex')
grid on

figure2 = figure('Color',[1 1 1]);
x=-1:.001:1;
E = f - (sin(pi*x)+cos(pi*x)+sin(pi*x).*cos(pi*x));
plot(x,E,'Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('Error','Interpreter','latex');
legend('$Error$','Interpreter','latex')
grid on

% 
% x=-1:.005:1;
% figure(2)
% plot(x,f,'r','linewidth',2)
% hold on
% plot(x,sin(pi*x)+cos(pi*x)+sin(pi*x).*cos(pi*x),'--b','linewidth',2)
% set(gca,'fontsize',11,'fontweight','bold')
% title('\bf \fontsize {13} Estimate: sin(\pix)+cos(\pix)+sin(\pix)cos(\pix) ')
% legend('\bf \fontsize {13} Estimate','\bf \fontsize {13} Target')
% grid on