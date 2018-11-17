prefix = "ILSVRC2012_val_0000000";
surfix = ".JPEG";

for i = 1:10%000
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
    x_bound = [max(0, floor(m/2)-50), min(m, floor(m/2)+49)];
    y_bound = [max(0, floor(n/2)-50), min(n, floor(n/2)+49)];
    x_idx = randi(x_bound, 100, 1);
    y_idx = randi(y_bound, 100, 1);
    for i = 1:100
       I_temp = I(x_idx(i):, y_idx(i), :);
       
    end
end