clc
clear
close all

alpha = -3;
beta = 3;
x = alpha:0.001:beta;

h = 1;
N = 7;

g_bar = zeros(N,1);
ej = zeros(N,1);

num = 0;
den = 0;

for j=1:N
    ej(j,1) = -3 + h*(j-1);
        if j == 1
            mu_A_x = trimf(x,[-3, -3, -3+h]);
        elseif j==N
            mu_A_x = trimf(x,[3-h, 3, 3]);
        else
            mu_A_x = trimf(x,[-3+h*(j-2), -3+h*(j-1), -3+h*(j)]);
        end

    g_bar(j,1) = sin(ej(j,1));
    num = num + g_bar(j,1).*mu_A_x;
    den=den+mu_A_x;
end

f_x = num./den;
g_x = sin(x);

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