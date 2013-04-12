#pragma once
#ifndef SIM_TEST_H
#define SIM_TEST_H
#include "../common.h"
#if IS_TESTING
#include <vector>
#include <cstdint>

std::vector<SimBody> SimHostTest(const std::vector<SimBody>& bodies,   uint32_t num_samples);
std::vector<SimBody> SimDeviceTest(const std::vector<SimBody>& bodies, uint32_t num_samples);

bool SimTest(uint32_t num_bodies, uint32_t samples, float* percentage);
void SimFullTest(uint32_t extra_passes);
#endif
#endif //SIM_TEST_H