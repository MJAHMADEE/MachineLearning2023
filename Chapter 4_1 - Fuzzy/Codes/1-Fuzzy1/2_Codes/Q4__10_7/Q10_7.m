clc
clear
close all

alpha = -1;
beta = 1;
x1 = alpha:0.001:beta;
x2 = alpha:0.001:beta;

% g_x = 0.52 + 0.1*x1 + 0.28*x2 - 0.06*x1.*x2;
g_dot_x1 = 0.1 - 0.06*x2;
g_dot_x2 = 0.28 - 0.06*x1;
norm_g_dot_x1 = norm(g_dot_x1,inf);
norm_g_dot_x2 = norm(g_dot_x2,inf);

epsilon = 0.1;
h = epsilon/(norm_g_dot_x1 + norm_g_dot_x2);
n = round((beta-alpha)/h);
N = n+1;

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


       g_bar(k+1,1) = 0.52 + 0.1*e_i1(i1,1) + 0.28*e_i2(i2,1) - 0.06*e_i1(i1,1)*e_i2(i2,1);
       num = num + g_bar(k+1,1).*mu_A_x1.*mu_A_x2;
       den=den+mu_A_x1.*mu_A_x2;
       k=k+1;
       end
end

f_x = num./den;
g_x = 0.52 + 0.1*x1 + 0.28*x2 - 0.06*x1.*x2;


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