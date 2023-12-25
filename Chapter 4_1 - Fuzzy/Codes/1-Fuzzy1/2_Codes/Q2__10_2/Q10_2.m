clc
clear
close all

alpha = -1;
beta = 1;
x = alpha:0.001:beta;

h = 0.01;
N = 201;

g_bar = zeros(N,1);
ej = zeros(N,1);

num = 0;
den = 0;

for j=1:N
    ej(j,1) = -1 + h*(j-1);
        if j == 1
            mu_A_x = trimf(x,[-1, -1, -1+h]);
        elseif j==N
            mu_A_x = trimf(x,[1-h, 1, 1]);
        else
            mu_A_x = trimf(x,[-1+h*(j-2), -1+h*(j-1), -1+h*(j)]);
        end

    g_bar(j,1) = sin(ej(j,1).*pi)+cos(ej(j,1).*pi)+sin(ej(j,1).*pi).*cos(ej(j,1).*pi);
    num = num + g_bar(j,1).*mu_A_x;
    den=den+mu_A_x;
end

f_x = num./den;
g_x = sin(x.*pi)+cos(x.*pi)+sin(x.*pi).*cos(x.*pi);

figure0 = figure('Color',[1 1 1]);
plot(x, g_x, 'b-', x, f_x, 'r--','Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('$g(x),f(x)$','Interpreter','latex');
legend('$g(x)$','$f(x)$','Interpreter','latex')
grid on

figure1 = figure('Color',[1 1 1]);
plot(x,f_x,'Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('f(x)','Interpreter','latex');
legend('$f(x)$','Interpreter','latex')
grid on

figure2 = figure('Color',[1 1 1]);
E = f_x - g_x;
plot(x,E,'Linewidth',2);
xlabel('x','Interpreter','latex');
ylabel('Error','Interpreter','latex');
legend('$Error$','Interpreter','latex')
grid on