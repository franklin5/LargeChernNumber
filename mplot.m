clear
clc
clf
EK = load('spectrum_synthetic.OUT');
NKX = sqrt(length(EK(:,1)));
kmax = 2;
NK = (NKX-1)/2;
akx = -kmax:2*kmax/(NKX-1):kmax;aky = akx;
[AKX,AKY]=meshgrid(akx,aky);
cutoffP = length(EK(1,:));
energy = zeros(NKX,NKX,cutoffP);
for nk = 0:NKX*NKX-1
    nkx = mod(nk,NKX);
    nky = floor(nk/NKX);
    energy(nkx+1,nky+1,:) = EK(nk+1,:);
end
Tperiod = 26.24;
figure(1)
for np = 1:cutoffP
    temp(:,:) = energy(:,:,np)/(pi/Tperiod);
    mesh(AKX,AKY,temp)
    hold on
end
hold off
xlabel('k_x/k_F')
ylabel('k_y/k_F')
zlabel('E(k_x,k_y)/(pi/T)')
title('hi=2.1,hf=0.9')
view(3)
axis([-kmax kmax -kmax kmax -5 5])
view([0 0])
%axis auto

% clear
% clc
% figure(3)
% Delta = load('test_Delta.OUT'); 
% T = 26.24;
% t= 0:0.0005:T;
% plot(t,Delta(:,1),'r',t,Delta(:,2),'b')
% legend('real','imag')
% hold on
% plot(t,0.2*cos(2*pi/T*t),'m')
% hold off