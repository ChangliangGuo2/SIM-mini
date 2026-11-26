function SRdata = HiFiSIM_stacks(data, system, iteration, otf, Wiener, ApoFWHM)
warning off;
param.nrDirs = system.nrDirs;
param.nrPhases = system.nrPhases;
param.phaOff=0;
param.nrBands=2;
param.FlagApoFWHM=0;
param.Size1 = size(data,1);
param.Size2 = size(data,2);
Mean = mean(data(:));
disp(Mean);
%% Generate approximate OTF/PSF
Iraw = double(data);% 读取数据并转化为双精度
NPixel = size(Iraw,1);
param.micronsPerPixel = system.pixelsize;
param.cyclesPerMicron = 1/(NPixel*param.micronsPerPixel);
param.NA = system.NA;
param.lambda = system.wavelength;
param.imgSize = NPixel;

param.attStrength = otf.attStrength;
param.OtfProvider=SimOtfProvider(param,param.NA,param.lambda,1);

psf=abs(otf2psf((param.OtfProvider.otf)));
psfatt=abs(otf2psf((param.OtfProvider.otfatt)));

handles.param.OtfProvider=param.OtfProvider;
handles.param=param;

%% Preprocessing
N=param.nrDirs*param.nrPhases;
Temp=importImages(Iraw);
% imWS(Temp,['result/Raw data/双色/','Temp']);
if iteration == 0
    IIraw = Temp;
else
    IIraw=deconvlucy(Temp, psf, iteration);
end
% imWS(IIraw,['result/Raw data/双色/','IIraw']);
% IIraw = Temp;
for I=1:N
    IIrawFFT(:,:,I)=FFT2D(IIraw(:,:,I),false);
end

%% Parameter Estimation

% Kc region MASK
cnt=[NPixel/2+1,NPixel/2+1];
param.cutoff=1000/(0.5*param.lambda/param.NA);
[x,y]=meshgrid(1:NPixel,1:NPixel);
rad=sqrt((y-cnt(1)).^2+(x-cnt(2)).^2);
Mask=double(rad<=1.0*(param.cutoff/param.cyclesPerMicron+1));
NotchFilter0=getotfAtt(NPixel,param.OtfProvider.cyclesPerMicron,0.2*param.cutoff,0,0);
NotchFilter=NotchFilter0.*Mask;
Mask2=double(rad<=1.10*(param.cutoff/param.cyclesPerMicron+1));
NotchFilter2=NotchFilter0.*Mask2;

CrossCorrelation=zeros(size(Mask2,1),size(Mask2,2),param.nrDirs);
k0=zeros(1,param.nrDirs);

for I=1:param.nrDirs
    lb=2;
    if param.nrBands==2
        hb=2;
        fb=lb;
    elseif param.nrBands==3
        hb=4;
        fb=hb;
    end
    
    param.phaOff=0;
    param.fac=ones(1,param.nrBands);
    
    %     Temp=importImages(double(datalowSNR));
    %     IIraw=deconvlucy(Temp, psf, 5);
    %     % IIraw = Temp;
    %     for I=1:N
    %         IIrawFFT(:,:,I)=FFT2D(IIraw(:,:,I),false);
    %     end
    separateII=separateBands(IIrawFFT(:,:,(I-1) * param.nrPhases+1:I*param.nrPhases),param.phaOff,param.nrBands,param.fac, system.phase_shift);

    SeparateII{1,I}=separateII;
    
    if param.nrBands==2
        c0=separateII(:,:,1);
        c2=separateII(:,:,lb);
        
        c0=(c0./(max(max(abs(c0)))));
        c2=(c2./(max(max(abs(c2)))));

        c0=c0.*NotchFilter;
        c2=c2.*NotchFilter;
        
        c0=FFT2D(c0,false);
        c2=FFT2D(c2,false);
        c2=c2.*conj(c0);
        c2=c2./max(max(c2));
        
        vec=fftshift(FFT2D(c2,true));
    elseif param.nrBands==3
        c0=separateII(:,:,1);
        c3=separateII(:,:,hb);
        
        c0=c0./(max(max(abs(separateII(:,:,1))))).*NotchFilter;
        c3=c3./(max(max(abs(separateII(:,:,hb))))).*NotchFilter;
        
        c0=FFT2D(c0,false);
        c3=FFT2D(c3,false);
        c3=c3.*conj(c0);
        c3=c3./max(max(c3));
        
        vec=fftshift(FFT2D(c3,true));
    end
    CrossCorrelation(:,:,I)=vec;
    %%
    %         temp=vec.*NotchFilter;
    temp=vec.*NotchFilter2;
    temp=log(1+abs(temp));
    temp=temp./max(max(temp));
    %         MIJ.createImage(temp);
    
    [yPos,xPos]=find(temp==max(max(temp)));
    peak.xPos=xPos(1);
    peak.yPos=yPos(1);
    k0(I)=sqrt((peak.xPos-cnt(1))^2+(peak.yPos-cnt(2))^2);
end

Flag=0;
if param.nrDirs>2           % For very few special cases
    if max(k0)-min(k0)>8
        Flag=1;
        Kobject=min(k0);
        % Kobject=208;
        Mask1=rad>=(Kobject+1);
        Mask2=rad<=(Kobject-1);
    end
end

for I=1:param.nrDirs
    vec=CrossCorrelation(:,:,I);
    if Flag==1
        vec(Mask1)=0;
        vec(Mask2)=0;
    end
    temp=vec.*NotchFilter2;
    temp=log(1+abs(temp));
    temp=temp./max(max(temp));
    %         MIJ.createImage(temp);
    
    [yPos,xPos]=find(temp==max(max(temp)));
    peak.xPos=xPos(1);
    peak.yPos=yPos(1);
    
    cntrl=zeros(10,30);
    overlap=0.15;
    step=2.5;
    bn1=(param.nrBands-1)*2;
    kx=(peak.xPos-cnt(2));
    ky=(peak.yPos-cnt(1));
    
    separateII=SeparateII{1,I};
    [peak,~]=fitPeak(separateII(:,:,1)/(max(max(abs(separateII(:,:,1))))),separateII(:,:,fb)/(max(max(abs(separateII(:,:,fb))))),1,bn1,param.OtfProvider,-kx,-ky,overlap,step,cntrl);
    
    if lb~=hb
        if param.nrBands==2
            peak.kx=peak.kx*2;
            peak.ky=peak.ky*2;
        end
        
        p1=getPeak(separateII(:,:,1),separateII(:,:,lb),0,1,param.OtfProvider,peak.kx/2,peak.ky/2,overlap);
        p2=getPeak(separateII(:,:,1)/(max(max(abs(separateII(:,:,1))))),separateII(:,:,hb)/(max(max(abs(separateII(:,:,1))))),0,2,param.OtfProvider,peak.kx,peak.ky,overlap);
        
        param.Dir(I).px=-peak.kx/2;
        param.Dir(I).py=-peak.ky/2;
        param.Dir(I).phaOff=-phase(p1);
        Temp_m1=abs(p1);
        Temp_m2=abs(p2);
        
        Temp_m1(Temp_m1>1.0)=1;
        Temp_m2(Temp_m2>1.0)=1.0;
        param.Dir(I).modul(1)=Temp_m1;
        param.Dir(I).modul(2)=Temp_m2;
    end
    if lb==hb
        p1=getPeak(separateII(:,:,1),separateII(:,:,lb),1,lb,param.OtfProvider,peak.kx,peak.ky,overlap);
        param.Dir(I).px=-peak.kx;
        param.Dir(I).py=-peak.ky;
        param.Dir(I).phaOff=-phase(p1);
        Temp_m=abs(p1);
        Temp_m(Temp_m>1.0)=1.0;
        param.Dir(I).modul=Temp_m;
    end
    K0(I)=sqrt((param.Dir(I).px)^2+(param.Dir(I).py)^2);
end

%%
SIMparam=zeros(3,5);
if param.nrPhases==3
    for i=1:param.nrDirs
        SIMparam(i,1)=atan(param.Dir(i).py/param.Dir(i).px)*180/pi;
        SIMparam(i,2)=sqrt((param.Dir(i).px)^2+(param.Dir(i).py)^2);
        SIMparam(i,3)=param.Dir(i).phaOff;
        SIMparam(i,4)=param.Dir(i).modul;
        SIMparam(i,4)
        if param.Dir(i).modul<0.35
            param.Dir(i).modul=0.7;
        end
    end
elseif param.nrPhases==5
    for i=1:param.nrDirs
        SIMparam(i,1)=atan(param.Dir(i).py/param.Dir(i).px)*180/pi;
        SIMparam(i,2)=sqrt((param.Dir(i).px)^2+(param.Dir(i).py)^2)*2;
        SIMparam(i,3)=param.Dir(i).phaOff;
        SIMparam(i,4)=param.Dir(i).modul(1);
        SIMparam(i,5)=param.Dir(i).modul(2);
        if param.Dir(i).modul(1)<0.35
            param.Dir(i).modul(1)=0.7;
        end
        if param.Dir(i).modul(2)<0.35
            param.Dir(i).modul(2)=0.7;
        end
    end
end

handles.K0=K0;
handles.IIrawFFT=IIrawFFT;
handles.param.Dir=param.Dir;
handles.param=param;

IIrawFFT = handles.IIrawFFT;
Iraw = data;
siz=size(Iraw(:,:,1));
w=siz(2);
h=siz(1);


param.attStrength = otf.attStrength;
param.a = otf.belta;           % damping factor：belta
param.attFWHM = otf.attFWHM;
param.OtfProvider=SimOtfProvider(param,param.NA,param.lambda,param.a);
handles.param=param;

%% Spectrum optimization
fftDirectlyCombined=zeros(h*2,w*2);
for I=1:param.nrDirs
    par=param.Dir(I);
    param.fac(2:param.nrBands)=param.Dir(I).modul(1:param.nrBands-1);
    param.fac(2:param.nrBands)=param.Dir(I).modul(1:param.nrBands-1);
    separate=separateBands(IIrawFFT(:,:,(I-1)*param.nrPhases+1:I*param.nrPhases),par.phaOff,param.nrBands,param.fac, system.phase_shift);
    
    shifted=zeros(2*h,2*w,param.nrPhases);
    shifted(:,:,1)=placeFreq(separate(:,:,1));
    
    for b=2:param.nrBands
        pos=b*2-2;
        neg=b*2-1;
        shifted(:,:,pos)=placeFreq(separate(:,:,pos));
        shifted(:,:,neg)=placeFreq(separate(:,:,neg));
        
        shifted(:,:,pos)=NfourierShift(shifted(:,:,pos),-(b-1)*par.px,-(b-1)*par.py);
        shifted(:,:,neg)=NfourierShift(shifted(:,:,neg),(b-1)*par.px,(b-1)*par.py);
    end
    shifted(:,:,1)=applyOtf(shifted(:,:,1),param.OtfProvider,1,0,0,1,0);
    for b=2:param.nrBands
        pos=b*2-2;
        neg=b*2-1;
        shifted(:,:,pos)=applyOtf(shifted(:,:,pos),param.OtfProvider,b,-(b-1)*par.px,-(b-1)*par.py,1,0);
        shifted(:,:,neg)=applyOtf(shifted(:,:,neg),param.OtfProvider,b,(b-1)*par.px,(b-1)*par.py,1,0);
    end
    for J=1:param.nrBands*2-1
        fftDirectlyCombined=fftDirectlyCombined+shifted(:,:,J);
    end
end
% save OtfProvider-1.mat fftDirectlyCombined shifted

% Temp1=real(ifft2(fftshift((fftDirectlyCombined))));
% Temp1(Temp1<0)=0;
% imwrite(Temp1./max(Temp1(:)),'Temp1.tif')
% MIJ.createImage(Temp1);

w1=Wiener.w1;  % Initial optimization Wiener constant
w2=Wiener.w2;
%%
param.cutoff=1000/(0.5*param.lambda/param.NA);
param.sampleLateral=ceil(param.cutoff/param.cyclesPerMicron)+1;
K0=handles.K0;
K=max([ceil(K0)]);
if param.nrBands==2
    cutoff=floor(1*K)/param.sampleLateral+1.0;
    R=K;
elseif	param.nrBands==3
    cutoff=floor(2*K)/param.sampleLateral+1.0;
    R=2*K;
end
otfHiFi=zeros(2*h,2*w);
otfHiFi=writeApoVector(otfHiFi,param.OtfProvider,cutoff);      % Ideal OTF
Mask=zeros(2*h,2*w);
Mask(otfHiFi~=0)=1;

% %% Traditional Wiener-SIM
% if size(Iraw,3)==9
%     wFilter0=WienerFilterWiener_3D(param);
% else
%     wFilter0=WienerFilterWiener_3D(param);
% end
% Wk0=otfHiFi./(wFilter0.wDenom+w2^2);
% fftWiener=real(ifft2(fftshift((fftDirectlyCombined.*Wk0.*Mask))));
% Temp=fftWiener;
% Temp(Temp<0)=0;
% Temp=255*Temp/max(max(Temp));
% Wiener(1:Size1,1:Size2)=Temp(1:Size1,1:Size2);
% SRdata = importImages2(Wiener);

%% HiFi-Wiener
% Step 1
% if size(Iraw,3)==9
if param.nrBands==2
    wFilter1=WienerFilterW1_2D(param);
else
    wFilter1=WienerFilterW1_3D(param);
end

Wk1=otfHiFi./(wFilter1.wDenom+w1^2);

fftInitialHiFi=fftDirectlyCombined.*Wk1.*Mask;
% Temp3=real(ifft2(fftshift((fftInitialHiFi))));
% Temp3(Temp3<0)=0;
% MIJ.createImage(Temp3);

% Step 2
if size(Iraw,3)==9
    wFilter2=WienerFilterW2_2D(param);
else
    wFilter2=WienerFilterW2_3D(param);
end

if nargin < 6 || isempty(ApoFWHM)
    ApoFWHM=0.5*(cutoff-1);
    ApoFWHM=min(0.5,round(ApoFWHM*100)/100);
end
% end
apo= apodize_gauss([2*h,2*w], struct('rad',ApoFWHM));
Wk2=apo./(wFilter2.wDenom+w2^2);
fftHiFi=real(ifft2(fftshift((fftInitialHiFi.*Wk2.*Mask))));

% Results of HiFi-SIM
Size1=2*param.Size1;
Size2=2*param.Size2;
HiFi=zeros(Size1,Size2);
Temp=fftHiFi;
Temp(Temp<0)=0;
% Temp=255*Temp/max(max(Temp));
% Temp=Mean*Temp/max(max(Temp));
HiFi(1:Size1,1:Size2)=Temp(1:Size1,1:Size2);
SRdata = importImages2(HiFi);
SRdata = Mean*SRdata./max(SRdata(:));
