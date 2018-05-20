#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include </usr/local/cuda-8.0/include/cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 227;
static const int INPUT_W = 227;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 200;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "fc-rcnn";
const std::vector<std::string> directories{ "data/samples/faster-rcnn/", "data/faster-rcnn/" };
std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};
// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile, directories).c_str(),
															  locateFile(modelFile, directories).c_str(),
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));//data
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));//output

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{       const int N = 10;
	// create a GIE model from the caffe model and serialize it to a stream
    IHostMemory *gieModelStream{nullptr};
   	caffeToGIEModel("deploy.prototxt", "original.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, N, gieModelStream);
        
	// read a random sample image
	srand(unsigned(time(nullptr)));
	// available images 
	std::vector<std::string> imageList = { "1.ppm","2.ppm","3.ppm","4.ppm","5.ppm","6.ppm","7.ppm","8.ppm","9.ppm","10.ppm" };
	std::vector<PPM> ppms(N);

	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	// pixel mean used by the Faster R-CNN's author
	//float pixelMean[3]{ -1000.9801f, -1000.9465f, -1000.7717f }; // also in BGR order
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
        
	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[N*OUTPUT_SIZE];
        clock_t start = clock();
	doInference(*context, data, prob, N);
        clock_t ends=clock();
	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
        printf("%f",(double)1000*(ends-start)/CLOCKS_PER_SEC);
	// print a histogram of the output distribution
	for (int  i = 0; i<200; i++){
            //if(prob[i]>0.0)
                printf("%f\n", prob[i]);
        }
}
