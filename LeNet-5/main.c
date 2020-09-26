#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define FILE_TRAIN_IMAGE		"train-images.idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels.idx1-ubyte"
//#define FILE_TEST_IMAGE		"TestingData/t10k-images.idx3-ubyte"
//#define FILE_TEST_LABEL		"TestingData/t10k-labels.idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
//#define COUNT_TEST		10000

#define FILE_TEST_IMAGE		"TestingData/demopics.idx3-ubyte"  //Must add directory, file is in a subfolder
#define FILE_TEST_LABEL		"TestingData/demolabel.idx1-ubyte" //Must add directory, file is in a subfolder
#define COUNT_TEST			20

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	//fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}


int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		printf("label: %u\n", l);
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		//if (i * 100 / total_size > percent)
			//printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}


int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}



void foo()
{
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();

	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));
	
	free(lenet);
	
	free(test_data);
	free(test_label);
	system("pause");
}

int main()
{
	foo();
	return 0;
}