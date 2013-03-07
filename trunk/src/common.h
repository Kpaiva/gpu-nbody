//Team Cosmosis

#pragma once
#ifndef _COMMON_H_
#define _COMMON_H_


//#define USE_SSE

#define IS_SIMULATION 0

//Amount of samples before average is output on screen.
#define SAMPLE_RATE 1000
//Time to run the simulation in seconds.
#define TIME_TO_LIVE 240
//Whether or not to use double precision.
#define USE_DOUBLE_PRECISION 0

#include <cstdlib>
#include <ctime>

template <typename T>
class _SimBody;

template <typename T>
class _Body;

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
typedef _Body<double> Body;
static double random(double min, double max) { return (min + (rand() % (int)(max - min + 1))); }
#else
typedef vec2<float> vec2_t;
typedef _SimBody<float> SimBody;
typedef _Body<float> Body;
static float random(float min, float max) { return (min + (rand() % (int)(max - min + 1))); }
#endif

#endif //_COMMON_H_