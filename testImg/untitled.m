clc;clear;close all
prefix = "ILSVRC2012_val_0000";
surfix = ".JPEG";

for i = 2%:10%000
    filename = strcat(prefix, num2str(i, '%04d'), surfix);
    I = imread(filename);
    [m, n, ~] = size(I);
    [x_idx, y_idx] = findSaliency(I);
    
    I_crop = I(x_idx:x_idx+63,y_idx:y_idx+63,:);
    figure, imshow(I_crop);
    mask = ones(m, n);
    bw = activatecontour(I_crop, mask);
    figure, imshow(bw);
%     img1 = imgaussfilt(I_crop, 2);
% 	img2 = imgaussfilt(I_crop, 2);
end

function [x_idx, y_idx] = findSaliency(I)
    [m, n, ~] = size(I);
    x_bound = [max(0, floor(m/2)-100), min(m, floor(m/2)+99)];
    y_bound = [max(0, floor(n/2)-100), min(n, floor(n/2)+99)];
    tempx = randi(x_bound, 5000, 1);
    tempy = randi(y_bound, 5000, 1);
    max_diff = 0;
    x_idx = 0;
    y_idx = 0;
    for i = 1:5000
       I_temp = I(tempx(i):min(tempx(i)+63, m), tempy(i):min(tempy(i)+63, n), :);
       diff_total = mean(diff(I_temp, 1, 1).^2, 'all') + mean(diff(I_temp, 1, 2).^2, 'all');
       if diff_total > max_diff
           x_idx = tempx(i);
           y_idx = tempy(i);
       end
    end
end

