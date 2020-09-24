close all;clear;clc;
names = {'3' '9p' '4' '5' '1p' '7' '6p' '5p' '2' '3p' '8p' '0' '6' '2p' '4p' '8' '0p' '7' '1' '9'};
HandDigits = imread(strcat('TestingData\AllDigits\',names{1},'.png'));
for i = 2:20
    temp = imread(strcat('TestingData\AllDigits\',names{i},'.png'));
    HandDigits = cat(2,HandDigits,temp);
end

combinedbw = HandDigits(:,:,1);
figure
imshow(combinedbw)

fid = fopen('TestingData\demopics.idx3-ubyte', 'wb');
COUNT = fwrite(fid, combinedbw, 'uint8', 'b');