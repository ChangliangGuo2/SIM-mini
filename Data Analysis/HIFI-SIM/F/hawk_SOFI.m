function [output1, output2] = hawk_SOFI(data, order)
    cdfile = cd;
    data = single(data);
    %% sofi2
    data3 = abs(data - 1 * mean(data,3));
    sofi2 = mean(data3(:,:,1:end-1).*data3(:,:,2:end),3);
    output1 = sofi2;
    %% hawk
    [data_x] = single(hawk_2D(data));
    cd(cdfile);
    data = data_x(2:end,2:end,:);
    % sofi4
    data2 = gpuArray(data);
    data = abs(data2 - 1 * mean(data2,3));
    data3 = abs(cumulant0(data, order));
%     data3 = abs(-mean(data(:,:,1:end-3).*data(:,:,4:end),3).*mean(data(:,:,2:end-2).*data(:,:,3:end-1),3)...
%                -mean(data(:,:,1:end-3).*data(:,:,3:end-1),3).*mean(data(:,:,2:end-2).*data(:,:,4:end),3)...
%                -mean(data(:,:,1:end-3).*data(:,:,2:end-2),3).*mean(data(:,:,3:end-1).*data(:,:,4:end),3)...
%                +mean(data(:,:,1:end-3).*data(:,:,2:end-2).*data(:,:,3:end-1).*data(:,:,4:end),3));
    sofi = gather(real(data3));
    output2(2:size(sofi,1)+1,2:size(sofi,2)+1) = sofi;
end