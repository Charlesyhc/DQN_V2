clear;
clc;

load data4band.mat
loss1=data4band(:,1);
loss2=data4band(:,2);
reward=data4band(:,3);


h_av=ones(1,1000)/1000;
temp=filter(h_av,1,reward);
plot(temp(1000:80000));




