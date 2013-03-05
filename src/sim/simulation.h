//Team Cosmosis

#pragma once
#ifndef SIMULATION_H
#define SIMULATION_H

#include <thrust/device_vector.h>
#include "simbody.cu"
#include "../common.h"

class Simulation {
private:
	thrust::device_vector<SimBody> bodies_;
	bool running_;
	int sampleCount_;
	unsigned numBlocks_;
	unsigned numThreads_;

	Simulation(void);
public:
	static Simulation& GetInstance(void);

	int Setup(int argc, char* argv[]);
	int Run(void);
};

#endif //SIMULATION_H
