function [ratio,density,bright]=getParameters(data,norm,timelag,weight)
tic
disp(['The molecular parameter extraction start...'])
data = abs(data-1 * mean(data,3));
for i = 2 : 4
if timelag == 0
    cum = cumulant_0timelag(data,i);%sofi-2nd
else 
    cum = cumulant0(data,i);
end
if norm == 1
    cum = cum/max(cum(:));
end
eval(['cum',num2str(i),'=cum;']);
end

cond = 2;
k1 = cum3./cum2/(3/2)^cond;%等式右侧的kappa2、3、4就是2、3、4阶SOFI，前面对于图像的重采样不必要整理出来，那个应该是跟时空交叉累积量相关的（有空西看看）
k2 = cum4./cum2/(4/2)^cond;
bright = sqrt(max(0,3*k1.^2 - 2*k2));
ratio = max(0,min(0.5-0.5*k1./bright,1));
density = max(0,cum2./(bright.^2.*ratio.*(1-ratio)));

% Weighted average using the 2nd-order cumulant as weight.
if weight==1
cum2=abs(cum2);
k1=1./sum(cum2,3);
ratio=sum(cum2.*ratio,3).*k1;
density=sum(cum2.*density,3).*k1;
bright=sum(cum2.*bright,3).*k1;
end


ttime = toc;
disp('The molecular parameter extraction completed.');
disp(['Total time cost: ', num2str(ttime/60) ' mins'])
