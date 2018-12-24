clc; clear; close all;

for idx = 7%1:4000
    prefix = num2str(idx, '%04d');
%     I1 = imread(strcat(prefix, '_1.jpeg'));
%     I2 = imread(strcat(prefix, '_2.jpeg'));
%     I0 = imread(strcat(prefix, '_0.jpeg'));
    I1 = imread('t3a.jpg');
    I2 = imread('t3b.jpg');
    
    [m, n, ~] = size(I1);
    rawMask1 = zeros(m, n);
    rawMask2 = zeros(m, n);
    
    blockSize = 4;
    for i = 1:m/blockSize
        for j = 1:n/blockSize
            xrange = ((i-1)*blockSize+1):i*blockSize;
            yrange = ((j-1)*blockSize+1):j*blockSize;
            
            block1 = I1(xrange, yrange, :);
            block2 = I2(xrange, yrange, :);
            
            rawMask1(xrange, yrange) = repmat(std2(block1), [blockSize, blockSize]);
            rawMask2(xrange, yrange) = repmat(std2(block2), [blockSize, blockSize]);
            
        end
    end
    montage({I1, I2, rawMask1 > rawMask2})
    
    %imwrite(rawMask1 > rawMask2, strcat(prefix, '_4.jpeg'));
end




