clear
clc
clf
EK = load('minEg.OUT');
figure(1)
set(gca,'fontsize',16)
plot(0.9:0.01:1.2,EK,'-o','linewidth',2)
xlabel('h/E_F')
ylabel('min|E_{k=0}|/E_F')
axis([0.9 1.25 0 0.12])