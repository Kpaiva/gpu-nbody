//Team Cosmosis

#pragma once
#ifndef _COMMON_H_
#define _COMMON_H_

//Whether or not to test the kernel with valid host code
#define IS_TESTING 0
//Determines if in simulation, non simulation NOT supported.
#define IS_SIMULATION 1
//Amount of samples before average is output on screen.
#define SAMPLE_RATE 1000
//Time to run the simulation in seconds.
#define TIME_TO_LIVE 60
//Whether or not to use double precision.
#define USE_DOUBLE_PRECISION 0

#if IS_TESTING
#define EPSILON (1.0)
#define ACCURACY (0.99)
#define USE_DOUBLE_PRECISION 1
#endif

#include <cstdlib>
#include <ctime>
#include <cmath>

template <typename T>
class _SimBody;

#ifdef __GNUC__
#define IS_LINUX 1
#else
#define IS_LINUX 0
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <typename T>
struct vec2
{
    T x, y;
    CUDA_CALLABLE_MEMBER vec2(void) : x(0.0f), y(0.0f) { }
    CUDA_CALLABLE_MEMBER vec2(T _x, T _y) : x(_x), y(_y) { }
};

#if USE_DOUBLE_PRECISION
typedef vec2<double> vec2_t;
typedef _SimBody<double> SimBody;
static double random(double min, double max) { return (min + (rand() % (int)(max - min + 1))); }
#else
typedef vec2<float> vec2_t;
typedef _SimBody<float> SimBody;
static float random(float min, float max) {
	return min + (floorf((float(rand())/RAND_MAX) * 10)/10) * (max-min);
}
#endif

#endif //_COMMON_H_
