#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H

#include <iostream>
#include <vector>
#include "simbody.cu"
#include "simulation.h"
#include "simtester.h"
#if IS_TESTING
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <stdint.h>

uint64_t SimHostTest(const std::vector<_SimBody<float>>& bodies, uint32_t num_samples)
{
	std::vector<_SimBody<float>> bodies_ = bodies;
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

	//Compute the checksum
	uint64_t checksum = 0;
	for(i = 0; i < bodies_.size(); ++i) {		
		checksum += (uint32_t)bodies_[i].Position.x;
		checksum += (uint32_t)bodies_[i].Position.y;
	}
	return checksum;
}

uint64_t SimDeviceTest(const std::vector<_SimBody<float>>& bodies, uint32_t num_samples)
{
	thrust::device_vector<_SimBody<float>> d_bodies;
	thrust::host_vector<_SimBody<float>> h_bodies;
	float timeStep = 25000.0f;

	d_bodies = bodies;

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
		SimCalc <<< blocks, threads>>>(arr);
		//Ensure that we have done all calculations before we move on to tick.
		cudaThreadSynchronize();

		//SimTick <<<numBlocks_, threads>>>(arr, timeStep);
		SimTick <<< blocks, threads>>>(arr, timeStep);
		//Ensure that we have ticked all before we move to calculate the average.
		cudaThreadSynchronize();
	}

	h_bodies = d_bodies;

	//Compute the checksum.
	uint64_t checksum = 0;
	for(unsigned i = 0; i < h_bodies.size(); ++i) {
		checksum += (uint32_t)h_bodies[i].Position.x;
		checksum += (uint32_t)h_bodies[i].Position.y;
	}
	return checksum;
}

bool SimTest(uint32_t num_bodies, uint32_t samples)
{
	std::vector<_SimBody<float>> bodies;
	uint32_t seed = (uint32_t)time(NULL);
	srand(seed);
	
	bodies.reserve(num_bodies);

    for (uint32_t i = 0; i < num_bodies; ++i) {
		bodies.push_back(_SimBody<float>(
						random(-16384.0f, 16384.0f),
						random(-16384.0f, 16384.0f),
						random(-16384.0f, 16384.0f),
						random(-16384.0f, 16384.0f),
						random(-16384.0f, 16384.0f)));
	}

	uint64_t host_ans = 0, device_ans = 0, diff = 0;

	host_ans = SimHostTest(bodies, samples);
	device_ans = SimDeviceTest(bodies, samples);

	diff = host_ans-device_ans;

	return diff == 0;
}

void SimFullTest(uint32_t extra_passes) 
{
	uint32_t success = 0;
	auto do_test = [](uint32_t n, uint32_t s)->bool{
		std::cout << "Testing " << n << " bodies (" << s << " samples)...";
		bool answer = SimTest(n, s);
		std::cout << (answer ? " passed." : " failed.") << std::endl;
		return answer;
	};

	success += !do_test(100, 128);
	success += !do_test(200, 128);
	success += !do_test(100, 64);

	for(uint32_t i(0); i != extra_passes; ++i) {
		success += !do_test((uint32_t)random(50,250),(uint32_t)random(64,128));
	}

	if(success == 0) 
		std::cout << "All tests passed!" << std::endl;	
	else 
		std::cout << success << " tests failed!" << std::endl;	
}
#endif IS_TESTING
#endif //SIM_TEST_H