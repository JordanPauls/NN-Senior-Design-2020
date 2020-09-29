close all;clear;clc;
names = {'3' '9p' '4' '5' '1p' '7' '6p' '5p' '2' '3p' '8p' '0' '6' '2p' '4p' '8' '0p' '7p' '1' '9'};
HandDigits = imread(strcat('TestingData\AllDigits\',names{1},'.png'));
for i = 2:20
    temp = imread(strcat('TestingData\AllDigits\',names{i},'.png'));
    HandDigits = cat(1,HandDigits,temp);
end

combinedbw = HandDigits(:,:,1);
combinedbwNew = 255-combinedbw;

figure
subplot(1,2,1)
imshow(combinedbw)
title('Raw')
subplot(1,2,2)
imshow(combinedbwNew)
title('Colors Inverted')

fid = fopen('TestingData\demopics.idx3-ubyte', 'wb');
COUNT = fwrite(fid, combinedbwNew', 'uint8', 'b');  %transpose before sending in binary!!!