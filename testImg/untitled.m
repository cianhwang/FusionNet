clc;clear;close all
addpath('~/Downloads/ILSVRC2012_img_val/')
prefix = "ILSVRC2012_val_0000";
surfix = ".JPEG";

for i = 1001:2000
    filename = strcat(prefix, num2str(i, '%04d'), surfix);
    I = imread(filename);
    [m, n, ~] = size(I);
    [x_idx, y_idx] = findSaliency(I);
    
    I_crop = I(x_idx:x_idx+63,y_idx:y_idx+63,:);
 %   figure, imshow(I_crop);
    %mask = randMask(I_crop);
%     mask = ones(64, 64);
%     bw = activecontour(I_crop, mask, 100);
%  %   figure, imshow(bw);
%     se = strel('disk',10);
%     openBW = imopen(bw,se);
%     %figure, imshow(openBW);
%     openBW = repmat(uint8(openBW), [1, 1, 3]);
%     
%     GaussianBlurParam1 = (rand*2+1);
%     GaussianBlurParam2 = (rand*2+1);
%     
%     img1 = imgaussfilt(I_crop.*openBW, GaussianBlurParam1) + I_crop.*(1-openBW);
%  %   figure, imshow(img1);
%  	img2 = imgaussfilt(I_crop.*(1-openBW), GaussianBlurParam2) + I_crop.*openBW;
%   %  figure, imshow(img2);
%     %figure, imshowpair(img1,img2, 'montage');
     imwrite(I_crop, strcat('~/Downloads/clearImg/',num2str(i, '%04d'), '_0.jpeg'));
%     imwrite(img1, strcat(num2str(i, '%04d'), '_1.jpeg'));
%     imwrite(img2, strcat(num2str(i, '%04d'), '_2.jpeg'));
end

function [x_idx, y_idx] = findSaliency(I)
    [m, n, ~] = size(I);
    x_bound = [max(0, floor(m/2)-100), min(m, floor(m/2)+99)];
    y_bound = [max(0, floor(n/2)-100), min(n, floor(n/2)+99)];
    tempx = randi(x_bound, 100, 1);
    tempy = randi(y_bound, 100, 1);
    max_diff = 0;
    x_idx = 0;
    y_idx = 0;
    for i = 1:100
        try
       I_temp = I(tempx(i):tempx(i)+63, tempy(i):tempy(i)+63, :);
       diff_total = mean(diff(I_temp, 1, 1).^2, 'all') + mean(diff(I_temp, 1, 2).^2, 'all');
       if diff_total > max_diff
           x_idx = tempx(i);
           y_idx = tempy(i);
       end
        catch
            %disp(i);
            continue;
        end
    end
end

