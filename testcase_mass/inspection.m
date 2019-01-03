clc;clear;close all
for i = 1:20000
    I = imread(strcat(num2str(i, '%05d'), '_2.jpeg'));
    disp(i);
    [m, n, dim] = size(I);
    assert(m==n);
    assert(m==72);
    assert(dim==3);
%     if dim == 1
%         imwrite(repmat(I, [1, 1, 3]), strcat(num2str(i, '%05d'), '_0.jpeg'));
% %         I1 = imread(strcat(num2str(i, '%05d'), '_1.jpeg'));
% %         I2 = imread(strcat(num2str(i, '%05d'), '_2.jpeg'));
% %         imwrite(repmat(I1, [1, 1, 3]), strcat(num2str(i, '%05d'), '_1.jpeg'));
% %         imwrite(repmat(I2, [1, 1, 3]), strcat(num2str(i, '%05d'), '_2.jpeg'));
%     end
end