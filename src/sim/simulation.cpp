//Coded by Clinton Bale
//02/06/2013

#include "simulation.h"
#include "../timer.h"

#include <iostream>

Simulation::Simulation(void) : sampleCount_(-1) { }
Simulation::~Simulation(void) { }

Simulation& Simulation::GetInstance() {
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
	if(num_bodies < 0 || num_bodies > 100000) {
		std::cout << "** Invalid number of bodies, must be between 1 and 100000. **" << std::endl;
		return 1;
	}
	std::cout << "Setting up " << num_bodies << " bodies." << std::endl;
	srand(time(NULL));
	bodies_.reserve(num_bodies);
	for(unsigned i = 0; i < num_bodies; ++i)
		bodies_.push_back(SimBody(
			random(1.0E11,3.0E11),
			random(-6.0E11,9.0E11),
			random(-1000.0,1000.0),
			random(-1000.0,1000.0),
			random(1.0E9, 1.0E31)));
	std::cout << "Completed setup... computing... " << std::endl;
	return 0;
}

void Simulation::Tick(double dt) {
	size_t i = 0;
	size_t j = 0;
	
	unsigned x = 0;
	for(i = 0; i < bodies_.size(); ++i) {
		bodies_[i].ResetForce();

		for(j = 0; j < bodies_.size(); ++j) {
			if(i != j) { 
				bodies_[i].AddForce(bodies_[j]);
			}
		}
	}
	
	for(i = 0; i < bodies_.size(); ++i) {		
		bodies_[i].Tick(dt);		
	}
}

int Simulation::Run(void)
{
	running_ = true;
	double timeStep = 25000.0f;
	
	double average = 0.0f;
	unsigned sample = 0;

	Timer timer;
	timer.start();
	if(sampleCount_ > 0)
		std::cout << "Running test for " << sampleCount_ << " samples..." << std::endl;
	while(running_) {
		Tick(timeStep);
		++sample;	

		if(sampleCount_ < 0) {
			if(sample % SAMPLE_RATE == 0) {
				std::cout << sample << " Samples taken avg. " << std::fixed 
				<< double(timer.getElapsedTimeInMilliSec() / (double)sample) 
				<< " ms/sample. " << double((double)sample / timer.getElapsedTimeInSec())
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
		<< double(timer.getElapsedTimeInMilliSec() / (double)sample) 
		<< " ms/sample. " << double((double)sample / timer.getElapsedTimeInSec())
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
