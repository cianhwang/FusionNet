% prepare images
foreground = zeros(90,160,3, 'single');
foreground(21:70, 51:70, 1) = 1;
foreground(21:70, 71:90, 2) = 1;
foreground(21:70, 91:115, 3) = 1;
foreground([1:20,71:90],:,:) = 0.5;
foreground(:,[1:50,116:160],:) = 0.5;
alpha = zeros(90,160, 'single');
alpha(21:70, 51:115) = 1; %alpha should be a 0/1 single image
background = ones(90, 160, 3, 'single')*0.5;
background(1:45, 1:80, :) = 1;
background(46:90, 81:160, :) = 1;

% background = imgaussfilt(background, 3) .* alpha +background;


% figure();
% subplot(2,2,1);
% imshow(foreground);
% title('foreground image');
% subplot(2,2,2);
% imshow(alpha);
% title('foreground mask');
% background(31:60, 61:105, :) = 0;
% 
% subplot(2,2,4);
% imshow(background);
% title('background image(fill boundary)');
% % background(21:70, 51:115, :) = 0;
% subplot(2,2,3);
% imshow(background);
% title('background image');


% all-in-focus merge
aif_merged = merge_image(foreground, alpha, 0.1, background);
figure();
imshow(aif_merged);
title('all-in-focus merged image');
% figure();
% % far-focused merge, blur foreground_image*alpha
% ff_merged = merge_image(foreground, alpha, 5, background);
% subplot(2,2,2);
% imshow(ff_merged);
% title('far-focus merged image (improved)');
% 
% % near-focused merge, blur background
% nf_merged = merge_image(foreground, alpha, 0.01, imgaussfilt(background,5));
% subplot(2,2,4);
% imshow(nf_merged);
% title('near-focus merged image (improved)');
% 
% I = aif_merged;
% 
% aa = imgaussfilt(I,5).* alpha + I.*(1-alpha);
% bb = imgaussfilt(I,5).* (1-alpha) + I.*alpha;
% 
% subplot(2, 2, 1);
% imshow(aa);
% title('far-focus merged image');
% subplot(2, 2, 3);
% imshow(bb);
% title('near-focus merged image');
% 
% 
function merged = merge_image(foreground, alpha, blur_sigma, background)
% expand 1 channel foreground alpha to 3 channel
alpha = padarray(alpha,[0,0,2],'post','replicate');
% blur foreground_image*alpha
foreground_to_add = imgaussfilt(alpha.*foreground, blur_sigma);
% apply complementary blurred alpha on background
blurred_alpha = imgaussfilt(alpha, blur_sigma);
background_to_add = (1-blurred_alpha).*background;
% multiply and sum
merged = foreground_to_add + background_to_add;
end