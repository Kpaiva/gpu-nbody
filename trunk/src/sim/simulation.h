//Team Cosmosis

#pragma once
#ifndef SIMULATION_H
#define SIMULATION_H

#include <thrust/device_vector.h>
#include "simbody.cu"
#include "../common.h"

typedef struct
{
    SimBody *array;
    unsigned size;
} BodyArray;

BodyArray MakeArray(thrust::device_vector<SimBody> &arr);
void __global__ SimCalc(BodyArray a);
void __global__ SimTick(BodyArray a, float dt);

class Simulation {
private:
	thrust::device_vector<SimBody> bodies_;
	bool running_;
	int sampleCount_;
	unsigned numBlocks_;
	unsigned numThreads_;
	unsigned maxResidentThreads_;
	unsigned blocks_;

	Simulation(void);
public:
	static Simulation& GetInstance(void);

	int Setup(int argc, char* argv[]);
	int Run(void);
};

#endif //SIMULATION_H
