//Team Cosmosis

#include "simulation.h"
#include "simbody.cu"
#include "..\timer.h"

typedef struct {
	SimBody* array;
	unsigned size;
} BodyArray;

BodyArray MakeArray(thrust::device_vector<SimBody>& arr) {
	BodyArray ba = { thrust::raw_pointer_cast(&arr[0]), arr.size() };
	return ba;
}

void __global__ SimCalc(BodyArray a) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < a.size) {
		a.array[idx].ResetForce();

		for(size_t j = 0; j < a.size; ++j) {
			if(idx != j) { 
				a.array[idx].AddForce(a.array[j]);
			}
		}
	}
}

void __global__ SimTick(BodyArray a, float dt) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < a.size) {
		a.array[idx].Tick(dt);	
	}
}

Simulation::Simulation(void) : sampleCount_(-1), numBlocks_(0), numThreads_(0) { }

Simulation& Simulation::GetInstance(void) {
	static Simulation self;
	return self;
}

int Simulation::Setup(int argc, char* argv[]) {
	if(argc < 2) {
		std::cout << "Invalid number of arguments." << std::endl <<
			"Usage: " << argv[0] << " [num bodies] <max samples>" << std::endl;
		return 1;
	}
	if(argc == 3) {
		int do_samples = atoi(argv[2]);
		if(do_samples < 1 || do_samples > 10240) {
			std::cout << "** Invalid number of samples, must be between 1 and 10240. **" << std::endl;
			return 1;
		}
		sampleCount_ = do_samples;
	}
	int num_bodies = atoi(argv[1]);
	if(num_bodies < 0 || num_bodies > 16384) {
		std::cout << "** Invalid number of bodies, must be between 1 and 16384. **" << std::endl;
		return 1;
	}
	std::cout << "Setting up " << num_bodies << " bodies." << std::endl;
	srand(time(NULL));
	bodies_.reserve(num_bodies);
	for(unsigned i = 0; i < num_bodies; ++i)
		bodies_.push_back(SimBody(
		random(1.0E11f,3.0E11f),
		random(-6.0E11f,9.0E11f),
		random(-1000.0f,1000.0f),
		random(-1000.0f,1000.0f),
		random(1.0E9f, 1.0E31f)));
	std::cout << "Configuring CUDA... " << std::endl;

	int device;
	cudaDeviceProp prop;
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaGetDevice(&device);
	if(err != cudaSuccess) {
		std::cout << "Error getting CUDA device... aborting." << std::endl;
		return 1;
	}
	err = cudaGetDeviceProperties(&prop, device);
	if(err == cudaErrorInvalidDevice) {
		std::cout << "Invalid CUDA device found... aborting." << std::endl;
		return 1;
	}
	int max_threads = prop.maxThreadsDim[0];

	numBlocks_ = (bodies_.size() + max_threads - 1) / max_threads;
	numThreads_ = max_threads;

	std::cout << "CUDA setup complete. Using:" << std::endl <<
		"\tBlocks: " << numBlocks_ << std::endl <<
		"\tThreads: " << numThreads_ << std::endl;
	std::cout << "Completed setup... computing... " << std::endl;
	return 0;
}

int Simulation::Run(void) {
	running_ = true;
	float timeStep = 25000.0f;

	unsigned sample = 0;

	Timer timer;
	timer.start();
	if(sampleCount_ > 0)
		std::cout << "Running test for " << sampleCount_ << " samples..." << std::endl;

	BodyArray arr = MakeArray(bodies_);		
	while(running_) {
		SimCalc<<<numBlocks_, numThreads_>>>(arr);
		//Ensure that we have done all calculations before we move on to tick.
		cudaThreadSynchronize();

		SimTick<<<numBlocks_, numThreads_>>>(arr, timeStep);
		//Ensure that we have ticked all before we move to calculate the average.
		cudaThreadSynchronize();

		++sample;	

		if(sampleCount_ < 0) {
			if(sample % SAMPLE_RATE == 0) {
				std::cout << sample << " Samples taken avg. " << std::fixed 
					<< float(timer.getElapsedTimeInMilliSec() / (float)sample) 
					<< " ms/sample. " << float((float)sample / timer.getElapsedTimeInSec())
					<< " samples/sec. " << std::endl;
			}
			if(timer.getElapsedTimeInSec() > TIME_TO_LIVE)
				running_ = false;
		}
		else if(sample == sampleCount_) {
			running_ = false;	
		}
	}
	timer.stop();
	std::cout << sample << " Samples taken avg. " << std::fixed 
		<< float(timer.getElapsedTimeInMilliSec() / (float)sample) 
		<< " ms/sample. " << float((float)sample / timer.getElapsedTimeInSec())
		<< " samples/sec. " << std::endl;
	if(sampleCount_ > 0)
		std::cout << "Total elapsed time: " << timer.getElapsedTimeInSec() << " seconds." << std::endl;
	std::cout << "Completed the test with " << sample << " samples. Press any key to exit." << std::endl;
#if IS_LINUX
	std::cin >> sample;
#else
	system("pause");
#endif
	return 0;
}