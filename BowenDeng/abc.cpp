#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <string.h>  
#include <vector>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>           
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"   
#pragma warning (disable:4996)
using namespace std;
using namespace cv;
static const int TEST_SCALES = 600;
static const int TEST_MAX_SIZE = 1000;
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
typedef struct {
	int w;
	int h;
	int c;
	int step;
	float *data;
} image;
image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}
void ipl_into_image(IplImage* src, image im)
{
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;
	int i, j, k;

	for (i = 0; i < h; ++i) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < w; ++j) {
				im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
			}
		}
	}
}
image ipl_to_image(IplImage* src)
{
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	image out = make_image(w, h, c);
	ipl_into_image(src, out);
	out.step = src->widthStep;
	return out;
}
void image_into_ipl( image im, IplImage* dst)
{

	
}
static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}
void free_image(image m)
{
	if (m.data) {
		free(m.data);
	}
}

static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] += val;
}
image resize_image(int i, image im, int w, int h)
{
	image resized = make_image(w, h, im.c);
	resized.step = im.step;
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < im.h; ++r) {
			for (c = 0; c < w; ++c) {
				float val = 0;
				if (c == w - 1 || im.w == 1) {
					val = get_pixel(im, im.w - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1) continue;
			for (c = 0; c < w; ++c) {
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}
struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};
void foo(int i, const std::string& filename, PPM& ppm)
{
	//std::ifstream infile(locateFile(filename), std::ifstream::binary);
	//std::string imageinput = filepath+filename;
	//cv::Mat rawjpg = imread(filename, -1);
	const char *p = filename.c_str();
	cout << p;
	IplImage* imgSrc = cvLoadImage(p, -1);
	ppm.h = imgSrc->height;
	ppm.w = imgSrc->width;
	//cout << imgSrc->channelSeq[0] << imgSrc->channelSeq[1] << imgSrc->channelSeq[2] << endl;
	image tmp = ipl_to_image(imgSrc);
	image tt = resize_image(i, tmp, 224, 224);
	//for (i = 0; i < h; ++i) {
		//for (k = 0; k < c; ++k) {
			//for (j = 0; j < w; ++j) {
				//im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
	//Mat  mat = cvarrToMat(imgSrc);
	for (int j = 0; j < 224; j++) {
		for (int i = 0; i < 224; i++)
		{
			ppm.buffer[(j * 224 + i) * 3] = tt.data[3 * (j * 224 + i) + 2] * 255;
			if (tt.data[3 * (j * 224 + i) + 2] * 255 == 10)
			{
				cout << "Yes" << endl;
				cout << tt.data[3 * (j * 224 + i) + 2] * 255 << endl;
			}
			cout << ppm.buffer[(j * 224 + i) * 3] << endl;
			ppm.buffer[(j * 224 + i) * 3 + 1] = tt.data[(j * 224 + i) + 1] * 255;
			ppm.buffer[(j * 224 + i) * 3 + 2] = tt.data[(j * 224 + i) + 0] * 255;
		}
		//}
		//}
		/*imgSrc->imageData = (char*)calloc(224 * 224 * 3, sizeof( char));
		imgSrc->height = tt.h;
		imgSrc->width = tt.w;
		imgSrc->nChannels = tt.c;
		imgSrc->widthStep = 3 * 224;
		int m, j, k;
		for (m = 0; m < tt.h; ++m)
		{
			for (k = 0; k < tt.c; ++k)
			{
				for (j = 0; j < tt.w; ++j)
				{
					imgSrc->imageData[m * 3 * 224 + j * tt.c + k] = tt.data[k * tt.w * tt.h + m * tt.w + j] * 255;
					cout << "image to IPImage " << tt.data[k * tt.w * tt.h + m * tt.w + j] * 255 << endl;
					cout << "image to IPImage " << imgSrc->imageData[m * 3 * 224 + j * tt.c + k] << endl;
				}
			}
		}
		Mat jpg= cvarrToMat(imgSrc);
		for (int j = 0; j<jpg.rows; j++) {
			uint8_t *data = jpg.ptr<uchar>(j);
			for (int i = 0; i<jpg.cols; i++)
			{
				ppm.buffer[(j*jpg.rows + i) * 3] = data[3 * i + 2];
				ppm.buffer[(j*jpg.rows + i) * 3 + 1] = data[3 * i + 1];
				ppm.buffer[(j*jpg.rows + i) * 3 + 2] = data[3 * i + 0];
			}
		}
		free(imgSrc->imageData);
		cout << i << endl;
		*/
	}
}
char f(int i)
{
	if (i == 0)
		return '0';
	else if (i == 1)
		return '1';
	else if (i == 2)
		return '2';
	else if (i == 3)
		return '3';
	else if (i == 4)
		return '4';
	else if (i == 5)
		return '5';
	else if (i == 6)
		return '6';
	else if (i == 7)
		return '7';
	else if (i == 8)
		return '8';
	else if (i == 9)
		return '9';
}
int main()
{
	char name[] = "D://BaiduYunDownload/ILSVRC2013_DET_val/ILSVRC2013_val_00000001.JPEG";
	float j = 48.3;
	unsigned char i = j;
	cout << i << endl;
	cout << sizeof(unsigned char) << endl;
	/*for (int i = 4598; i < 4599; ++i)
	{
		name[59] = f(i / 1000);
		name[60] = f((i / 100) % 10);
		name[61] = f((i / 10) % 10);
		name[62] = f(i % 10);
		//int i = 2;
		std::vector<PPM> ppms(1);
		foo(i, name, ppms[0]);
		int b = 1;
	}
	*/
	return 0;
}