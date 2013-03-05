//Coded by Clinton Bale
//02/06/2013

#pragma once
#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include "simbody.h"
#include "../common.h"

class Simulation {
private:
	std::vector<SimBody> bodies_;
	bool running_;
	int sampleCount_;

	Simulation(void);
	~Simulation(void);
public:
	static Simulation& GetInstance();

	int Setup(int argc, char* argv[]);
	void Tick(double dt);
	int Run(void);	
};

#endif //SIMULATION_H
