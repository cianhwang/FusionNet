% prepare images
foreground = zeros(90,160,3, 'single');
foreground(21:70, 51:70, 1) = 1;
foreground(21:70, 71:90, 2) = 1;
foreground(21:70, 91:115, 3) = 1;
foreground([1:20,71:90],:,:) = 0.5;
foreground(:,[1:50,116:160],:) = 0.5;
alpha = zeros(90,160, 'single');
alpha(21:70, 51:115) = 1; %alpha should be a 0/1 single image
background = zeros(90, 160, 3, 'single');
background(1:45, 1:80, :) = 1;
background(46:90, 81:160, :) = 1;
figure();
subplot(3,1,1);
imshow(foreground);
title('foreground image');
subplot(3,1,2);
imshow(alpha);
title('foreground alpha');
subplot(3,1,3);
imshow(background);
title('background image');

figure();
% all-in-focus merge
aif_merged = merge_image(foreground, alpha, 0.1, background);
subplot(3,1,1);
imshow(aif_merged);
title('all-in-focus merged image');

% far-focused merge, blur foreground_image*alpha
ff_merged = merge_image(foreground, alpha, 5, background);
subplot(3,1,2);
imshow(ff_merged);
title('far-focus merged image');

% near-focused merge, blur background
nf_merged = merge_image(foreground, alpha, 0.1, imgaussfilt(background,5));
subplot(3,1,3);
imshow(nf_merged);
title('near-focus merged image');

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