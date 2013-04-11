#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H

#include <vector>
#include "../common.h"

float SimHostTest(const std::vector<_SimBody<float>>& bodies);
float SimDeviceTest(const std::vector<_SimBody<float>>& bodies);

bool SimTest(int num_bodies);
void SimFullTest(int passes);

#endif //SIM_TEST_H