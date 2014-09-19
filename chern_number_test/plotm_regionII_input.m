clear
clc
close all
idata = 1;
RFILENAME = ...
    {'Re_del_hi2.1hf1.2.dat'};
IFILENAME = ...
    {'Im_del_hi2.1hf1.2.dat'};

data = load(RFILENAME{idata}) +...
    1i * load(IFILENAME{idata});
len = length(data);
time = real(data(:,1));
ReDelta = real(data(:,2));
ImDelta = imag(data(:,2));
Delta = ReDelta+1i*ImDelta;
figure(idata)
plot(time, real(Delta), 'r', time, imag(Delta), 'b',...
    time, abs(Delta),'k','linewidth',2)
phaseD = phase(Delta);
tempindex = 2000;
%delta0 = mean(abs(Delta(40*tempindex:60*tempindex)))
%muInf = (phaseD(60*tempindex) -...
%            phaseD(40*tempindex))/(-2*time(tempindex*20))
%Tperiod = abs(pi/muInf)
Tperiod = 26.24;
muInf = pi/Tperiod;
dt = time(2)-time(1);
start_index = 50/dt;
end_index = start_index + Tperiod/dt;
t = time(start_index:end_index);
Rtemp = cos(2*muInf*t).*real(Delta(start_index:end_index))...
    - sin(2*muInf*t).*imag(Delta(start_index:end_index));
Itemp = sin(2*muInf*t).*real(Delta(start_index:end_index))...
    + cos(2*muInf*t).*imag(Delta(start_index:end_index));
figure(idata+1)
plot(t,Rtemp,'r',t,Itemp,'b',t,abs(Delta(start_index:end_index)),'k')
save('Rdata_2112.dat','Rtemp','-ascii', '-double')
save('Idata_2112.dat','Itemp','-ascii', '-double')