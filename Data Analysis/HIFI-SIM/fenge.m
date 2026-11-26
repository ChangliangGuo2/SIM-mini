clc;
close;
% 读取图像
grayImage = imread('test.tif');
% grayImage = rgb2gray(image); % 如果是彩色图像，转换为灰度图像


% % Otsu 分割
% level = graythresh(grayImage); % Otsu 阈值
% BW = imbinarize(grayImage, level); % 应用阈值进行二值化

% 预处理：二值化图像
threshold = graythresh(grayImage); % 自动确定阈值
BW = imbinarize(grayImage, threshold); % 二值化图像
% 形态学处理：去噪
BW = imopen(BW, strel('disk', 1)); % 去除小噪点

% 计算每个区域的属性
stats = regionprops(BW, 'Area', 'BoundingBox', 'Circularity', 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');

% 设置圆形率和长宽比阈值
circleThreshold = 0.4; % 圆形率阈值，低于此值的结构被认为不是圆形
aspectRatioThreshold = 1.2; % 长宽比阈值，排除过长的结构

% 创建一个用于绘制标记的图像副本
markedImage = grayImage; % 如果原图像是彩色，则可以直接复制

if size(markedImage, 3) == 1
    markedImage = cat(3, markedImage, markedImage, markedImage); % 如果是灰度图像，转换为 RGB
end
% 统计和标记块状结构
numBlocks = 0;

for k = 1:length(stats)
    % 提取当前区域的属性
    circularity = stats(k).Circularity; % 圆形率
    majorAxis = stats(k).MajorAxisLength; % 主轴长度
    minorAxis = stats(k).MinorAxisLength; % 副轴长度
    boundingBox = stats(k).BoundingBox;

    % 计算长宽比
    aspectRatio = majorAxis / minorAxis;

    % 判断是否为块状结构（圆形率低且长宽比接近1）
    if circularity > circleThreshold && aspectRatio < aspectRatioThreshold
        % 创建一个与原图像同大小的全零图像
        mask = false(size(BW));
        
        % 使用像素索引列表填充目标区域
        mask(stats(k).PixelIdxList) = true;
        
        % 将目标区域填充为蓝色
        markedImage(:,:,1) = markedImage(:,:,1); % 保持红色通道不变
        markedImage(:,:,2) = markedImage(:,:,2); % 保持绿色通道不变
        markedImage(:,:,3) = markedImage(:,:,3) .* uint8(~mask) + uint8(mask) * 255; % 设置蓝色通道
        
        numBlocks = numBlocks + 1; % 更新块状结构数量
    end
end
% 保存标记后的图像
imwrite(markedImage, 'marked_image_blue.tif'); % 保存为 TIFF 文件

% 显示结果
figure;
imshow(markedImage);
title('Detected Non-Circular Blocks (Blue Fill)');
disp(['Number of Block-shaped Structures: ', num2str(numBlocks)]);


