#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H

#include <iostream>
#include <vector>
#include "simbody.cu"
#include "simtester.h"

float SimHostTest(const std::vector<_SimBody<float>>& bodies)
{
	std::vector<_SimBody<float>> bodies_ = bodies;
	size_t i = 0;
	size_t j = 0;
	for(int sample = 0; sample < 128; ++sample){
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
	float checksum = 0.0f;
	for(i = 0; i < bodies_.size(); ++i) {		
			checksum += bodies_[i].Position.x + bodies_[i].Position.y;
	}
	return checksum;
}

float SimDeviceTest(const std::vector<_SimBody<float>>& bodies)
{
	return 1.0;
}

bool SimTest(int num_bodies)
{
	std::vector<_SimBody<float>> bodies;
	unsigned seed = unsigned(time(NULL));
	srand(seed);
	
	//std::cout << "Testing host/device with " << num_bodies << " bodies." << std::endl;	
	bodies.reserve(num_bodies);
	//std::cout << "Setting up bodies with seed " << seed << "... ";

    for (unsigned i = 0; i < num_bodies; ++i) {
		bodies.push_back(_SimBody<float>(
						random(1.0E11f, 3.0E11f),
						random(-6.0E11f, 9.0E11f),
						random(-1000.0f, 1000.0f),
						random(-1000.0f, 1000.0f),
						random(1.0E9f, 1.0E31f)));
	}
	//std::cout << "done." << std::endl;

	float host_ans = 0.0f, device_ans = 0.0f, diff = 0.0f;

	//std::cout << "Testing host ... ";
	host_ans = SimHostTest(bodies);
	//std::cout << "done." << std::endl;

	//std::cout << "Testing device ... ";
	device_ans = SimDeviceTest(bodies);
	//std::cout << "done." << std::endl;

	diff = abs(host_ans-device_ans);

	/*std::cout << "Results:" << std::endl <<
		"Host  : " << host_ans << std::endl <<
		"Device: " << device_ans << std::endl <<
		"Diff  : " << diff << std::endl <<
		"Pass  : " << (diff < 0.5f ? "true" : "false") << std::endl;*/

	return diff < 0.5f;
}

void SimFullTest(int passes) 
{
	#define TEST(x,i) { bool __ans = SimTest((x)); std::cout << "Test #" << (i) << (__ans ? " passed." : " failed.") << std::endl; }

	TEST(4,0);
	for (unsigned i = 1; i <= passes; ++i) {
		TEST(i*100,i);
	}

	#undef TEST
}

#endif //SIM_TEST_H