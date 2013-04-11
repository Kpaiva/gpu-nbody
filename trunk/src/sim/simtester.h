#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H

#include <vector>
#include <stdint.h>
#include "../common.h"

uint64_t SimHostTest(const std::vector<_SimBody<float>>& bodies,   uint32_t num_samples);
uint64_t SimDeviceTest(const std::vector<_SimBody<float>>& bodies, uint32_t num_samples);

bool SimTest(uint32_t num_bodies, uint32_t samples);
void SimFullTest(uint32_t extra_passes);

#endif //SIM_TEST_H