#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <string.h>  
#include <vector>
#include <unistd.h>  
#include <dirent.h>       
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>           
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"   
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
using namespace std;
using namespace cv;
static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 201;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";


const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };
const std::string filepath = "/data/faster-rcnn/Images_2018_ppm_reshape/";

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};

struct BBox
{
	float x1, y1, x2, y2;
};
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
image resize_image(image im, int w, int h)
{
	image resized = make_image(w, h, im.c);
	resized.step = im.c * w;
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
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/faster-rcnn/", "data/faster-rcnn/","/home/nvidia/workspace/Images_2018_ppm_reshape/","/home/nvidia/workspace/Images_2018/","/home/nvidia/workspace/ILSVRC2013_DET_val2/"};
    return locateFile(input, dirs);
}
void jpg2ppm(const std::string& filename, PPM& ppm)
{
  ppm.fileName = filename;
	IplImage* imgSrc = cvLoadImage(locateFile(filename).c_str(), -1);
	ppm.h = imgSrc->height;
	ppm.w = imgSrc->width;
	image tmp = ipl_to_image(imgSrc);
	int new_memory = -1;
	image resized = resize_image(tmp, INPUT_H,  INPUT_W);
	if (imgSrc->height * imgSrc->width < INPUT_H * INPUT_W)
	{
    cvReleaseData(imgSrc);
		imgSrc->imageData = (char*)calloc(INPUT_H * INPUT_W * INPUT_C, sizeof(char));
		new_memory = 1;
	}
	imgSrc->height = resized.h;
	imgSrc->width = resized.w;
	imgSrc->nChannels = resized.c;
	imgSrc->widthStep = resized.c * resized.w;
	int m,  j,  k;
	for (m = 0; m < resized.h; ++m)
	{
		for (k = 0; k < resized.c; ++k)
		{
			for (j = 0; j < resized.w; ++j)
			{
				imgSrc->imageData[m * imgSrc->widthStep + j * imgSrc->nChannels  + k] = resized.data[k * resized.w * resized.h + m * resized.w + j] * 255;
			}
		}
	}
	Mat jpg = cvarrToMat(imgSrc);
	for (int j = 0; j < jpg.rows; j++) {
		uint8_t *data = jpg.ptr<uchar>(j);
		for (int i = 0; i < jpg.cols; i++)
		{
			ppm.buffer[(j * jpg.rows + i) * 3] = data[3 * i + 2];
			ppm.buffer[(j * jpg.rows + i) * 3 + 1] = data[3 * i + 1];
			ppm.buffer[(j * jpg.rows + i) * 3 + 2] = data[3 * i + 0];
		}
	}
	if (new_memory == 1)
		free(imgSrc->imageData);
  free_image(tmp);
	free_image(resized);
  cvReleaseImage(&imgSrc);
	return;
}



// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM& ppm, const BBox& bbox)
{
	std::ofstream outfile("./" + filename, std::ofstream::binary);
	assert(!outfile.fail());
	outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
	auto round = [](float x)->int {return int(std::floor(x + 0.5f)); };
	for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
	{
		// bbox top border
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
		// bbox bottom border
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
	}
	for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
	{
		// bbox left border
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
		// bbox right border
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
	}
	outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}
//Read all the pics in one folder
std::vector<std::string> getFiles(std::string cate_dir)  
{  
    std::vector<std::string> files;   
    DIR *dir;  
    struct dirent *ptr;  
    //char base[1000];  
   
    if ((dir=opendir(cate_dir.c_str())) == NULL)  
        {  
        perror("Open dir error...");  
                exit(1);  
        }  
   
    while ((ptr=readdir(dir)) != NULL)  
    {  
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
                continue;  
        else if(ptr->d_type == 8)    ///file  
            files.push_back(ptr->d_name);  
        else if(ptr->d_type == 10)    ///link file  
            continue;  
        else if(ptr->d_type == 4)    ///dir  
        {  
            files.push_back(ptr->d_name);  
        }  
    }  
    closedir(dir);  
    return files;  
}  
void caffeToGIEModel(const std::string& deployFile,			// name for caffe prototxt
	const std::string& modelFile,			// name for model 
	const std::vector<std::string>& outputs,		// network outputs
	unsigned int maxBatchSize,				// batch size - NB must be at least as large as the batch we want to run with)
	nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
	IHostMemory **gieModelStream)			// output stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);

	std::cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		nvinfer1::DataType::kHALF);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(10 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	(*gieModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 3 outputs.
	assert(engine.getNbBindings() == 5);
	void* buffers[5];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
		outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);


	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // affeparser1::data
	CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[inputIndex1]));
	CHECK(cudaFree(buffers[outputIndex0]));
	CHECK(cudaFree(buffers[outputIndex1]));
	CHECK(cudaFree(buffers[outputIndex2]));
}

template<int OutC>
class Reshape : public IPlugin
{
public:
	Reshape() {}
	Reshape(const void* buffer, size_t size)
	{
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
		assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
		return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	
	size_t getSerializationSize() override
	{
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};


// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kHALF, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kHALF, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return (!strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused"));
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPluginRshp2.release();		mPluginRshp2 = nullptr;
		mPluginRshp18.release();	mPluginRshp18 = nullptr;
		mPluginRPROI.release();		mPluginRPROI = nullptr;
	}


	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};


void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}


int main(int argc, char** argv)
{
<<<<<<< HEAD
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 1;
        const int M = 4599;
	const char *cache_path = "/home/fanzc/TensorRT-3.0.4 (2)/data/faster-rcnn/engine";
	std::stringstream searilizedengine;
	caffeToGIEModel("faster_rcnn_test_iplugin.prototxt",
=======
		// create a GIE model from the caffe model and serialize it to a stream
		PluginFactory pluginFactory;
		IHostMemory *gieModelStream{ nullptr };
		// batch size
		const int N = 1;
        const int M = 20000;
		/*char *cache_path = "/home/fanzc/TensorRT-3.0.4 (2)/data/faster-rcnn/engine";
		std::stringstream searilizedengine;*/
		caffeToGIEModel("faster_rcnn_test_iplugin.prototxt",
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f
		"vgg16_faster_rcnn_iter_80000.caffemodel",
		std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2 },
		N, &pluginFactory, &gieModelStream);

<<<<<<< HEAD
	pluginFactory.destroyPlugin();
	std::vector<std::string> imageList=getFiles("/home/nvidia/workspace/Images_2018");
//	std::vector<std::string> imageList=getFiles("/home/nvidia/workspace/ILSVRC2013_DET_val2");
//        std::vector<std::string> imageList=getFiles("/home/nvidia/workspace/ILSVRC2013_DET_val2");
=======
		pluginFactory.destroyPlugin();
    	std::vector<std::string> imageList=getFiles("/home/nvidia/workspace/Images_2018");
        //std::vector<std::string> imageList=getFiles("/home/nvidia/workspace/ILSVRC2013_DET_val2");
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f
        std::sort(imageList.begin(),imageList.end());
        std::vector<PPM> ppms(N);
        // deserialize the engine 
<<<<<<< HEAD
	IRuntime* runtime = createInferRuntime(gLogger);
	/*searilizedengine.write((const char*)gieModelStream->data(), gieModelStream->size());
        std::ofstream outfile;
        outfile.open(cache_path);
        outfile << searilizedengine.rdbuf();
        outfile.close();*/
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
=======
		IRuntime* runtime = createInferRuntime(gLogger);
		/*searilizedengine.write((const char*)gieModelStream->data(), gieModelStream->size());
		std::ofstream outfile;
		outfile.open(cache_path);
		outfile << searilizedengine.rdbuf();
		outfile.close();*/
		ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f
/*
char *cache_path = "/home/fanzc/TensorRT-3.0.4 (2)/data/faster-rcnn/engine";
std::ifstream cache(cache_path);
std:stringstream giemodelstream;
giemodelstream.seekg(0,giemodelstream.beg);
giemodelstream<<cache.rdbuf();
giemodelstream.seekg(0,giemodelstream.end);
const int modelsize = giemodelstream.tellg();
giemodelstream.seekg(0,giemodelstream.beg);
char* modelmem = malloc(modelsize);
giemodelstream.read(modelmem,modelsize);
ICudaEngine* engine = runtime->deserializeCudaEngine(modelmem, modelsize, &pluginFactory);
*/
<<<<<<< HEAD
	IExecutionContext *context = engine->createExecutionContext();
	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
=======
		IExecutionContext *context = engine->createExecutionContext();
		float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f
	 // host memory for outputs 
        float* rois = new float[N * nmsMaxOut * 4];
        float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
        float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];

         // predicted bounding boxes
        float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];

<<<<<<< HEAD
	float imInfo[N * 3]; // input im_info	
	//std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(ppms.size() <= imageList.size());
	for(int pn = 0; pn<M; ++pn)
=======
		float imInfo[N * 3]; // input im_info	
		//std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
		assert(ppms.size() <= imageList.size());
		for(int pn = 0; pn<M; ++pn)
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f
        {
           stringstream stream;
           stream << pn;
		       string string_temp = stream.str();
                for (int i = 0; i < N; ++i)
	        {
		        //readPPMFile(imageList[i+pn], ppms[i]);
		        jpg2ppm(imageList[i+pn], ppms[i]);
		        imInfo[i * 3] = 224.0 ;   // number of rows
		        imInfo[i * 3 + 1] = 224.0 ; // number of columns
		        imInfo[i * 3 + 2] = 1;         // image scale
	        }
                
	        // pixel mean used by the Faster R-CNN's author
	        float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
	        for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
	        {
		        for (int c = 0; c < INPUT_C; ++c)
		        {
			        // the color image to input should be in BGR order
			        for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
			        	data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
		        }
	        }


        	// run inference
        	doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);

	

        	// unscale back to raw image space
        	for (int i = 0; i < N; ++i)
        	{
        		float * rois_offset = rois + i * nmsMaxOut * 4;
        		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
        			rois_offset[j] /= imInfo[i * 3 + 2];
        	}
        
        	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	        const float nms_threshold = 0.3f;
	        const float score_threshold = 0.8f;

	        for (int i = 0; i < N; ++i)
	        {
		        float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		        float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		        for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		        {
		        	std::vector<std::pair<float, int> > score_index;
		        	for (int r = 0; r < nmsMaxOut; ++r)
		        	{
			        	if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				        {
				        	score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
				        	std::stable_sort(score_index.begin(), score_index.end(),
				        		[](const std::pair<float, int>& pair1,
				        			const std::pair<float, int>& pair2) {
				        		return pair1.first > pair2.first;
				        	});
				        }
			        }

			        // apply NMS algorithm
			        std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			        // Show results
			        for (unsigned k = 0; k < indices.size(); ++k)
			        {
				        int idx = indices[k];
				        std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				        //std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					//<< " (Result stored in " << storeName << ")." << std::endl;

				        BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
                                
                                    std::cout  << string_temp << " " << c <<  " " <<  scores[idx*OUTPUT_CLS_SIZE + c]  <<  " " << b.x1*ppms[i].w/224  << " " <<  b.y1*ppms[i].h/224 << " " <<  b.x2*ppms[i].w/224 << " " <<  b.y2*ppms[i].h/224 << std::endl;
				        //writePPMFileWithBBox(storeName, ppms[i], b);
					std::ofstream result_file;
                                        result_file.open("submission.csv", std::ios_base::app);
<<<<<<< HEAD
                                        result_file << string_temp << " " << c <<  " " <<  scores[idx*OUTPUT_CLS_SIZE + c]  <<  " " << b.x1*ppms[i].w/224  << " " <<  b.y1*ppms[i].h/224 << " " <<  b.x2*ppms[i].w/224 << " " <<  b.y2*ppms[i].h/224 << std::endl;
=======
                                        result_file << ppms[i].fileName << " " << c <<  " " <<  scores[idx*OUTPUT_CLS_SIZE + c]  <<  " " << b.x1*ppms[i].w/224  << " " <<  b.y1*ppms[i].h/224 << " " <<  b.x2*ppms[i].w/224 << " " <<  b.y2*ppms[i].h/224 << std::endl;
>>>>>>> fc23ce7914e36072be9c4d3c2a016b65b394e18f

			        }
		        }
	        }
        }

	/*delete[] data;
	delete[] rois;
	delete[] bboxPreds;
	delete[] clsProbs;
	delete[] predBBoxes;*/
        context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();
	return 0;
}
