close all;clear;clc;
Digits = imread('TestingData\PencilDigits\0.png');
for i = 1:9
    temp = imread(strcat('TestingData\PencilDigits\',string(i),'.png'));
    Digits = cat(2,Digits,temp);
end

combinedbw = Digits(:,:,1);
% figure
% imshow(combinedbw)

% fid = fopen('demopics.idx3-ubyte', 'wb');
% COUNT = fwrite(fid, combinedbw, 'uint8', 'b');