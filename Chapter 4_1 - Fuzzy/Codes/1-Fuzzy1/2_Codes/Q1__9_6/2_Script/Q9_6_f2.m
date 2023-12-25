clc
clear
close all

x1 = -1:0.01:2;
x2 = -1:0.01:2;
[~,n1] = size(x1);
[~,n2] = size(x2);
f2_x = zeros(n1,n2);

for i = 1:n1
    for j = 1:n2
    mu_A1_x1 = (x1(i)>=-1 & x1(i)<=1).*(1-abs(x1(i)));
    mu_A1_x2 = (x2(i)>=-1 & x2(i)<=1).*(1-abs(x2(j)));
    mu_A2_x1 = (x1(i)>=0 & x1(i)<=2).*(1-abs(x1(i)-1));
    mu_A2_x2 = (x2(i)>=0 & x2(i)<=2).*(1-abs(x2(j)-1));
        P1 = mu_A1_x1 * mu_A2_x2;
        P2 = mu_A1_x2 * mu_A2_x1;
        f2_x(i,j) = P2/(P1+P2);
    end
end

[x1,x2] = meshgrid(x1,x2);
f2_x = transpose(f2_x);

figure1 = figure('Color',[1 1 1]);
plot3(x1,x2,f2_x,'b');
xlabel('$x1$','Interpreter','latex')
ylabel('$x2$','Interpreter','latex')
zlabel('$f_2(x)$','Interpreter','latex');
grid on