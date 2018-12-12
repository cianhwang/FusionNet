clc; clear; close all;
for idx = 1:4000
    prefix = num2str(idx, '%04d');
    I1 = imread(strcat(prefix, '_1.jpeg'));
    I2 = imread(strcat(prefix, '_2.jpeg'));
    I0 = imread(strcat(prefix, '_0.jpeg'));
    
    rawMask1 = zeros(48, 48);
    rawMask2 = zeros(48, 48);
    
    for i = 1:12
        for j = 1:12
            xrange = i*4-3:i*4;
            yrange = j*4-3:j*4;
            
            block1 = I1(xrange, yrange, :);
            block2 = I2(xrange, yrange, :);
            
            rawMask1(xrange, yrange) = repmat(std2(block1), [4, 4]);
            rawMask2(xrange, yrange) = repmat(std2(block2), [4, 4]);
            
        end
    end
    
    imwrite(rawMask1 > rawMask2, strcat(prefix, '_4.jpeg'));
end



