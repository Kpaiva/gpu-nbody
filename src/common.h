//Coded by Clinton Bale
//02/06/2013

#ifndef _COMMON_H_
#define _COMMON_H_

//#define USE_SSE
#define IS_SIMULATION 0
//Amount of samples before average is output on screen.
#define SAMPLE_RATE 1000
//Time to run the simulation in seconds.
#define TIME_TO_LIVE 240

#include <cstdlib>
#include <ctime>

#ifdef __GNUC__
#define IS_LINUX 1
#else
#define IS_LINUX 0
#endif

typedef struct {
	double x,y;
} vec2d_t;

static double random(double min, double max) {
	return min + (rand() % (int)(max - min + 1));
}

#endif //_COMMON_H_
