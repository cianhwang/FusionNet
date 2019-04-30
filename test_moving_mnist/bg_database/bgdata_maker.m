clc;clear;close all

prefix = 'clutter/Foliage/input/in';

for zzz= 150+(1:50)
group_idx = strcat('bg_database/', num2str(zzz), '_');
kkk = 0;
type = randi(8);
xlabel = 64 + randi(144-99);
ylabel = 64 + randi(200-99);
for fileIdx = 1
    I = imread('clutter/Foliage/input/in000000.jpg');
    I = I(xlabel-32:xlabel+31, ylabel-32:ylabel+31, :);
    I = dataAug(I, type);
%     figure,
%     imshow(I)
    imwrite(I, strcat(group_idx, num2str(kkk, '%03d'), '.jpg'));
end
for fileIdx = randi(350)+(1:20)
    filename = strcat(prefix, num2str(fileIdx, '%06d'), '.jpg');
    I = imread(filename);
    I = I(xlabel-32:xlabel+31, ylabel-32:ylabel+31, :);
    I = dataAug(I, type);
%     figure,
%     imshow(I)
    kkk = kkk + 1;
    imwrite(I, strcat(group_idx, num2str(kkk, '%03d'), '.jpg'));
end
end

function img = dataAug(img, type)
    switch type
        case 1
            img = rot90(img, 1);
        case 2
            img = rot90(img, 2);
        case 3
            img = rot90(img, 3);
        case 4
            img = img;
        case 5
            img = rot90(img(:, :, end:-1:1), 1);
        case 6
            img = rot90(img(:, :, end:-1:1), 2);
        case 7
            img = rot90(img(:, :, end:-1:1), 3);
        case 8
            img = img(:, :, end:-1:1);    
    end
end