clc;clear;close all;warning off;
addpath(genpath('F'));
file2='Stack2';
addfile = '';
file = [file2,addfile];
ex=exist([file,'.mat'],'file');
% if ex==2
    load([file,'.mat']);
%     data=imreadstack([file,'.tif']);
% else
%     save ([file,'.mat'], 'data','-v7.3')
% end
eval(['sofi_psf = ',file,';']);
for ii = 2:20
[sofi_psf(:,:,1),sofi_psf(:,:,ii)] = reg3D(sofi_psf(:,:,1),sofi_psf(:,:,ii));
end
%%
%sparse 
mu = 500;
sigma_t = 3;
l10 = 5;
backg = 0;%强去背景
backg2 = 0.9;%弱去背景
%sofi
skip = 20;
frame = size(sofi_psf,3)/skip;
frame = 1;
iter1 = 5;
iter2 = 5;
pixel = 60 * 10^-9;
finter = 2;
FWHM1 = 300 * 10^-9;
FWHM2 = 300 * 10^-9;
data=double(sofi_psf);
savedata=data(:,:,1:skip);
Ipsf1 = generate_psf(finter*FWHM1/pixel);
Ipsf2 = generate_psf(finter*FWHM2/pixel);
%% SOFI
for f=1:frame
    f
    datasub=data(:,:,(f-1)*skip+1:f*skip);
    datasubl=fourierInterpolation(datasub,[finter,finter,1],'lateral');
    datasubled = abs(datasubl-1 * mean(datasubl,3));

        WF = mean(mean(datasubl(:,:,1:end-1),3),3);
        WF_stack(:,:,f) = WF;

        SOFI_2nd = mean(datasubled(:,:,1:end-1).*datasubled(:,:,2:end),3);
        SOFI_2nd_stack(:,:,f) = SOFI_2nd;

%% SACD

  if backg~=0
        medVal = mean(datasub(:)).*backg;
        sub_temp = datasub;
        sub_temp(sub_temp > medVal)=medVal;
        
        backgrounds= background_estimation(sub_temp,1,4,'db3',3);
        backgrounds(backgrounds<0)=0;
        datasub=datasub-backgrounds;
        datasub(datasub<0)=0;
  end 
  if backg2~=0
        backgrounds= background_estimation(datasub.*backg2,1,7,'db6',3);
        backgrounds(backgrounds<0)=0;
        datasub=datasub-backgrounds;
        datasub(datasub<0)=0;
  end
  
    datadeconl=fourierInterpolation(datasub,[finter,finter,1],'lateral');
    datadeconl(datadeconl<0)=0;
    for i=1:size(datasub,3)
%         datadeconl(:,:,i)=deconvlucy(datadeconl(:,:,i),Ipsf1,iter1);
        datadeconl(:,:,i)=RL3D(datadeconl(:,:,i),Ipsf1,iter1,1);
    end
    
    datadeconl=abs(datadeconl-1*mean(datadeconl,3));
    
%     cum(2:N-1,2:N-1) = (mean(datadeconl(1:N-2,2:N-1,:).*datadeconl(3:N,2:N-1,:),3)+mean(datadeconl(2:N-1,1:N-2,:).*datadeconl(2:N-1,3:N,:),3))./2;
    cum = mean(datadeconl(:,:,1:end-1).*datadeconl(:,:,2:end),3);
    cum=abs(cum);
    cum=cum./max(cum(:));
    SACD_2nd_RL1_stack(:,:,f) = cum;
    sacd = double(deconvlucy(cum,Ipsf2.^2, iter2));
    sacd=sacd./max(sacd(:));
    SACD_2nd_RL2_stack(:,:,f) = sacd;
    
%     sparse = sparse_main(cum,mu,sigma_t,l10,100,0);
%     sSACD = double(deconvlucy(sparse,Ipsf3.^2, iter2));
%     sSACD = sSACD./max(sSACD(:));
%     sSACD_stack(:,:,f) = sSACD;

end
%     sparse = sparse_main(SACD_2nd_RL1_stack,mu,sigma_t,l10,100,0);
%     for aa = 1:size(sparse,3)
%     sssacd = double(deconvlucy(sparse(:,:,aa),Ipsf2.^2, iter2));
%     sssacd = sssacd./max(sssacd(:));
%     sssacd_stack(:,:,aa) = sssacd;
%     end


 %% save   
    mkdir(['results']);

    WF_stack = WF_stack./max(max(WF_stack(:)));
    imwritestack((WF_stack*65535),['results\WF.tif']);

    SOFI_2nd_stack = SOFI_2nd_stack./max(max(SOFI_2nd_stack(:)));
    imwritestack(((SOFI_2nd_stack.^0.5)*65535),['results\SOFI.tif']);

    SACD_2nd_RL2_stack = SACD_2nd_RL2_stack./max(max(SACD_2nd_RL2_stack(:)));
    imwritestack(((SACD_2nd_RL2_stack.^0.5)*65535),['results\SACD.tif']);
    
    
%     sSACD_stack = sSACD_stack./max(max(sSACD_stack(:)));
%     imwritestack(((sSACD_stack.^0.5)*65535),['results\sSACD_l.tif']);
    


