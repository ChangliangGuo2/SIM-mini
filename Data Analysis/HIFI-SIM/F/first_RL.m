function  [output] = first_RL(data, FWHM1, pixel, finter1, iter1, backg)
    data = gpuArray(single(data));
    data = data./max(max(data(:)));
    Ipsf1 = single(PSFget(pixel,finter1*FWHM1));
    % fourier
    data_fI = gather(abs(single(fourierInterpolation(data,[finter1,finter1,finter1],'both'))));
    clear data;
    % backfilter
    if backg ~= 0
    [data_fI] = BF(data_fI,backg);
    end
    % RL1
    data_fI = data_fI./max(max(data_fI(:)));
    output = gather(RL3D(data_fI,Ipsf1,iter1,1));
end
