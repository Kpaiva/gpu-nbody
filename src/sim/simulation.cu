//Team Cosmosis

#include "simulation.h"
#include "simbody.cu"
#include "..\timer.h"
#include <cuda_runtime.h>

#include <cstdint>

BodyArray MakeArray(thrust::device_vector<SimBody>& arr)
{
    BodyArray ba = { thrust::raw_pointer_cast(&arr[0]), arr.size() };
    return ba;
}

void __global__ SimCalc(BodyArray a)
{
    int_fast32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.size) {
		const _T G = 6.67384f * pow(10.0f, -11.0f);
		//precompute positions at index
		const _T px = a.array[idx].Position.x;
		const _T py = a.array[idx].Position.y;
		//mass at the index
		const _T M_idx = G*a.array[idx].Mass;

		a.array[idx].Force = vec2_t();
        for (int_fast32_t j(0); j != a.size; ++j) {
            if (idx != j) {
				_T dx = a.array[j].Position.x - px;
				_T dy = a.array[j].Position.y - py;
				_T r = sqrt(dx*dx + dy*dy);
				_T F = (M_idx*a.array[j].Mass)/(r*r);
				a.array[idx].Force.x += F * (dx / r);
				a.array[idx].Force.y += F * (dy / r);
            }
        }
    }
}

void __global__ SimTick(BodyArray a, _T dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.size)
    {
		_T mass = a.array[idx].Mass;
		a.array[idx].Velocity.x += dt * (a.array[idx].Force.x / mass);
		a.array[idx].Velocity.y += dt * (a.array[idx].Force.y / mass);
		a.array[idx].Position.x += dt * a.array[idx].Velocity.x;
		a.array[idx].Position.y += dt * a.array[idx].Velocity.y;
    }
}

Simulation::Simulation(void) : sampleCount_(-1), numBlocks_(0), numThreads_(0) { }

Simulation &Simulation::GetInstance(void)
{
    static Simulation self;
    return self;
}

int Simulation::Setup(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Invalid number of arguments." << std::endl <<
                  "Usage: " << argv[0] << " [num bodies] <max samples>" << std::endl;
        return 1;
    }
    if (argc == 3)
    {
        int do_samples = atoi(argv[2]);
        if (do_samples < 1 || do_samples > 10240)
        {
            std::cout << "** Invalid number of samples, must be between 1 and 10240. **" << std::endl;
            return 1;
        }
        sampleCount_ = do_samples;
    }
    int num_bodies = atoi(argv[1]);
    if (num_bodies < 0 || num_bodies > 16384)
    {
        std::cout << "** Invalid number of bodies, must be between 1 and 16384. **" << std::endl;
        return 1;
    }
    std::cout << "Setting up " << num_bodies << " bodies." << std::endl;
    srand((unsigned)time(NULL));

	bodies_.reserve(num_bodies);
    for (unsigned i = 0; i < (unsigned)num_bodies; ++i)
        bodies_.push_back(SimBody(
                              random(1.0E11f, 3.0E11f),
                              random(-6.0E11f, 9.0E11f),
                              random(-1000.0f, 1000.0f),
                              random(-1000.0f, 1000.0f),
                              random(1.0E9f, 1.0E31f)));
    std::cout << "Configuring CUDA... " << std::endl;

    int device;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        std::cout << "Error getting CUDA device... aborting." << std::endl;
        return 1;
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err == cudaErrorInvalidDevice)
    {
        std::cout << "Invalid CUDA device found... aborting." << std::endl;
        return 1;
    }
	int maxBlocks = prop.major > 2 ? 16 : 8;

	numThreads_ = prop.maxThreadsPerMultiProcessor / maxBlocks;
	numBlocks_ = (bodies_.size() + numThreads_ - 1) / numThreads_;

	numThreads_ = (numThreads_ + 1) & ~1;

    std::cout << "CUDA setup complete. Using:" << std::endl <<
              "\tBlocks: " << numBlocks_ << std::endl <<
              "\tThreads: " << numThreads_ << std::endl <<
			  "\tMax Blocks: " << ((prop.major > 2) ? 16 : 8) << std::endl <<
			  "\tMax Resident Threads: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Completed setup... computing... " << std::endl;
    return 0;
}

int Simulation::Run(void)
{
    running_ = true;
    float timeStep = 25000.0f;

    unsigned sample = 0;

    Timer timer;
    timer.start();
    if (sampleCount_ > 0)
        std::cout << "Running test for " << sampleCount_ << " samples..." << std::endl;

		
    BodyArray arr = MakeArray(bodies_);

    while (running_)
    {
		SimCalc <<<numBlocks_, numThreads_>>>(arr);
		//SimCalc <<<numBlocks_, numThreads_>>>(arr);
		cudaThreadSynchronize();
		SimTick <<<numBlocks_, numThreads_>>>(arr, timeStep);
		//SimTick <<<numBlocks_, numThreads_>>>(arr, timeStep);
        cudaThreadSynchronize();

        ++sample;

        if (sampleCount_ < 0)
        {
            if (sample % SAMPLE_RATE == 0)
            {
                std::cout << sample << " Samples taken avg. " << std::fixed
                          << float(timer.getElapsedTimeInMilliSec() / (float)sample)
                          << " ms/sample. " << float((float)sample / timer.getElapsedTimeInSec())
                          << " samples/sec. " << std::endl;
            }
            if (timer.getElapsedTimeInSec() > TIME_TO_LIVE)
                running_ = false;
        }
        else if (sample == sampleCount_)
        {
            running_ = false;
        }
    }
	cudaDeviceReset();	

    timer.stop();
    std::cout << sample << " Samples taken avg. " << std::fixed
              << float(timer.getElapsedTimeInMilliSec() / (float)sample)
              << " ms/sample. " << float((float)sample / timer.getElapsedTimeInSec())
              << " samples/sec. " << std::endl;
    if (sampleCount_ > 0)
        std::cout << "Total elapsed time: " << timer.getElapsedTimeInSec() << " seconds." << std::endl;
    std::cout << "Completed the test with " << sample << " samples. Press any key to exit." << std::endl;	
    return 0;
}