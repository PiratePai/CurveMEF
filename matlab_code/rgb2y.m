%% 两幅源图像都是彩色图像
clc;
clear all;
A_dir = "rgb_dir_path"; % 包含所有源图像A的文件夹路径
save_dir_A = "fused_y_dir_path/"; % 保存源图像A的Y通道的文件夹路径
fileFolder=fullfile(A_dir); 
dirOutput=dir(fullfile(fileFolder,'*')); % 匹配任意后缀的图像文件
fileNames = {dirOutput.name};
[m, num] = size(fileNames);
 if exist(save_dir_A,'dir')==0
  mkdir(save_dir_A);
 end
for i = 3:num
    
     name_A = fullfile(A_dir, fileNames{i});
    save_name_A = strcat(save_dir_A, fileNames{i});
    image_A = double(imread(name_A));
    [Y_A,Cb_A,Cr_A]=RGB2YCbCr(image_A); 
    [~,~,ext] = fileparts(fileNames{i}); % 获取文件后缀
    save_name_A = strcat(save_dir_A, fileNames{i});
    imwrite(uint8(Y_A), save_name_A);
    disp(save_name_A)
end
