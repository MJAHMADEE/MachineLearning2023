clc;
clear all;
close all;

load mgdata.dat
x = mgdata(:,2);
x0 = x(25:end);
x6 = x(19:end-6);
x12 = x(13:end-12);
x18 = x(7:end-18);
x24 = x(1:end-24);
data = [x24 x18 x12 x6 x0];

TrainData = data(1:800,:);
TestData = data(801:end,:);
