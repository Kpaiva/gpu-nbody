#Makefile for Linux version of Assignment 1
#Clinton Bale
#
#Used for simulating only and not for display.

GCC_VERSION = 4.7.1
PREFIX = /usr/local/gcc/${GCC_VERSION}/bin/
CC = ${PREFIX}gcc
CPP = ${PREFIX}g++

nbodysim: main.o
	$(CPP) -onbodysim main.o timer.o simulation.o simbody.o
	
main.o: main.cpp
	$(CPP) -c -Ofast -std=c++0x main.cpp timer.cpp sim/simbody.cpp sim/simulation.cpp

sse: mainsse.o
	$(CPP) -onbodysim_sse main.o timer.o simulation.o simbody.o

mainsse.o: main.cpp
	$(CPP) -c -Ofast -std=c++0x -march=native -mfpmath=sse main.cpp timer.cpp sim/simbody.cpp sim/simulation.cpp

clean:
	rm -f *.o nbodysim nbodysim_sse
