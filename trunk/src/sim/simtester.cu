#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H
#include "../common.h"
#if IS_TESTING
#include "simbody.cu"
#include "simulation.h"
#include "simtester.h"

#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdint>

std::vector<SimBody> SimHostTest(const std::vector<SimBody>& bodies, uint32_t num_samples)
{
	std::vector<SimBody> bodies_ = bodies;

	size_t i = 0;
	size_t j = 0;
	for(uint32_t sample(0); sample != num_samples; ++sample) {	
		for(i = 0; i < bodies_.size(); ++i) {
			bodies_[i].ResetForce();
			for(j = 0; j < bodies_.size(); ++j) {
				if(i != j) bodies_[i].AddForce(bodies_[j]);			
			}
		}
		for(i = 0; i < bodies_.size(); ++i) {		
			bodies_[i].Tick(25000.0f);		
		}
	}

	return bodies_;
}

std::vector<SimBody> SimDeviceTest(const std::vector<SimBody>& bodies, uint32_t num_samples)
{
	thrust::device_vector<SimBody> d_bodies;
	std::vector<SimBody> h_bodies(bodies.size());
	float timeStep = 25000.0f;

	d_bodies = bodies;

	int device;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << std::endl;
        return h_bodies;
    }
    err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        std::cout << "Error getting CUDA device... aborting." << std::endl;
        return h_bodies;
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err == cudaErrorInvalidDevice)
    {
        std::cout << "Invalid CUDA device found... aborting." << std::endl;
        return h_bodies;
    }
    int max_threads = prop.maxThreadsDim[0];

	int numBlocks_ = (d_bodies.size() + max_threads - 1) / max_threads;
	//int maxResidentThreads_ = max_threads;

	//int threads = maxResidentThreads_ / numBlocks_;
	BodyArray arr = MakeArray(d_bodies);

	//my stuff
	int threads;
    int numThreads = max_threads;
	int blocks;
	int maxResidentThreads_ = prop.maxThreadsPerMultiProcessor;

	prop.major > 2 ? blocks = 16 : blocks = 8;
	maxResidentThreads_ > numThreads ? threads = numThreads / blocks : threads = maxResidentThreads_ / blocks;
	////

	for(uint32_t sample(0); sample != num_samples; ++sample) {	
		//SimCalc <<<numBlocks_, threads>>>(arr);
		SimCalc <<< blocks, threads >>>(arr);
		//Ensure that we have done all calculations before we move on to tick.
		cudaThreadSynchronize();

		//SimTick <<<numBlocks_, threads>>>(arr, timeStep);
		SimTick <<< blocks, threads>>>(arr, timeStep);
		//Ensure that we have ticked all before we move to calculate the average.
		cudaThreadSynchronize();
	}

	//Copy the data back to the host.
	thrust::copy(d_bodies.begin(), d_bodies.end(), h_bodies.begin());
	return h_bodies;
}

bool SimTest(uint32_t num_bodies, uint32_t samples, float* percentage)
{
	std::vector<SimBody> bodies;
	uint32_t seed = (uint32_t)time(NULL);
	srand(seed);
	
	if(percentage) *percentage = 0.f;
	bodies.reserve(num_bodies);

    for (uint32_t i = 0; i < num_bodies; ++i) {
		bodies.push_back(SimBody(
							random(1.0E11f, 3.0E11f),
							random(-6.0E11f, 9.0E11f),
							random(-1000.0f, 1000.0f),
							random(-1000.0f, 1000.0f),
							random(1.0E9f, 1.0E24f)));
	}
	
	std::vector<SimBody> device;
	std::vector<SimBody> host;

	device = SimDeviceTest(bodies, samples);
	host   = SimHostTest(bodies, samples);
	
	if(device.size() != num_bodies ||
	   host.size() != num_bodies)
		return false;

	unsigned equal = 0;
	auto d_it = device.begin();
	auto h_it = host.begin();
	while(d_it != device.end() && h_it != host.end()) {
		if((*d_it)==(*h_it)) ++equal;
		++d_it; ++h_it;
	}

	float cmp = float(equal) / num_bodies;
	if(percentage) *percentage = cmp;
	return cmp > ACCURACY;
}

void SimFullTest(uint32_t extra_passes) 
{
	uint32_t success = 0;
	auto do_test = [](uint32_t n, uint32_t s)->bool{
		std::cout << "Testing " << n << " bodies (" << s << " samples)...";
		float acc;
		bool answer = SimTest(n, s, &acc);
		std::cout << std::setprecision(2) << std::fixed << (answer ? " passed. (" : " failed. (") << (acc*100.0f) << "%)" << std::endl;
		return answer;
	};

	success += !do_test(100, 128);
	success += !do_test(200, 128);
	success += !do_test(300, 512);

	for(uint32_t i(0); i != extra_passes; ++i) {
		success += !do_test((uint32_t)random(50,250),(uint32_t)random(64,128));
	}

	success += !do_test(100, 1024);

	if(success == 0) 
		std::cout << "All tests passed!" << std::endl;	
	else 
		std::cout << success << " tests failed!" << std::endl;	
}
#endif //IS_TESTING
#endif //SIM_TEST_H