clear
clc
%load regionI_del_hi2.1hf0.6.mat
%load regionI_del_hi2.1hf0.9.mat
load regionI_del_hi2.1hf1.2.mat
%load regionI_del_hi2.1hf1.5.mat
figure(idata)
set(gca,'fontsize',16)
j1 = 50;
j2 = 450;
for j = 1:nbrEigenvalues
    if j >j1 && j<j2
        if j == nbrEigenvalues/2 || j == nbrEigenvalues/2+1
    plot(akx, (bdgE(:,j))/(pi^2/Tperiod),'k--','linewidth',2)
        else
            plot(akx, (bdgE(:,j))/(pi^2/Tperiod),...
        'Color',[(j-j1)/(j2-j1) 0 1-(j-j1)/(j2-j1)])
        end
    end
    hold on
end
hold off
box on
% title(DATAFILENAME{idata})
xlabel('k_y/k_F')
ylabel('\epsilon/(\pi/T)')
axis([-0.1 0.1 -0.05 0.05])
