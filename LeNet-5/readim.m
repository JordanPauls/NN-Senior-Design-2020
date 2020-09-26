%read in idx3 images
fid = fopen('TestingData\t10k-images-16bytegone.idx3-ubyte');
A = fread(fid,[28 280],'uint8','b');
A = uint8(A);

figure 
subplot(1,2,1)
imshow(A)
title('Raw Encoded MNIST Images')
subplot(1,2,2)
imshow(A')
title('Transposed Images')
