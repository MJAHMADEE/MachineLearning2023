clc
clear all
close all
tic

alpha = -1;
beta = 1;
x1 = alpha:0.001:beta;
x2 = alpha:0.001:beta;
h = 0.25;
N = 9;

g_bar = zeros(N*N,1);
e_i1 = zeros(N,1);
e_i2 = zeros(N,1);

[x1,x2] = meshgrid(x1,x2);

num = 0;
den = 0;
k = 0;

for i1=1:N
    for i2=1:N
    e_i1(i1,1) = -1 + h*(i1-1);
    e_i2(i2,1) = -1 + h*(i2-1);
        if i1==1
            mu_A_x1 = trimf(x1, [-1,-1,-1+h]);
        elseif i1==N
            mu_A_x1 = trimf(x1,[1-h, 1, 1]);
        else
            mu_A_x1 = trimf(x1,[-1+h*(i1-2), -1+h*(i1-1), -1+h*(i1)]);
        end

        if i2==1
            mu_A_x2 = trimf(x2, [-1,-1,-1+h]);
        elseif i2==N
            mu_A_x2 = trimf(x2,[1-h, 1, 1]);
        else
            mu_A_x2 = trimf(x2,[-1+h*(i2-2), -1+h*(i2-1), -1+h*(i2)]);
        end


       g_bar(k+1,1) = 1./(1+e_i1(i1,1).^2+e_i2(i2,1).^2);
       num = num + g_bar(k+1,1).*mu_A_x1.*mu_A_x2;
       den=den+mu_A_x1.*mu_A_x2;
       k=k+1;
       end
end

f_x = num./den;
g_x = 1./(1+x1.^2+x2.^2);


figure1 = figure('Color',[1 1 1]);
mesh(x1,x2,f_x,'Linewidth',2);
xlabel('x1','Interpreter','latex');
ylabel('x2','Interpreter','latex');
zlabel('f(x)','Interpreter','latex');
legend('$f(x)$','Interpreter','latex')
grid on

figure2 = figure('Color',[1 1 1]);
E = g_x - f_x;
mesh(x1,x2,E,'Linewidth',2);
xlabel('x1','Interpreter','latex');
ylabel('x2','Interpreter','latex');
zlabel('Error','Interpreter','latex');
legend('$Error$','Interpreter','latex')
grid on

toc

