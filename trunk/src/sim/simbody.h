//Coded by Clinton Bale
//02/06/2013

#pragma once
#ifndef BODY_H
#define BODY_H

#include "../common.h"

class SimBody {
private:
	double		mass_;

	inline double ComputeGC(double m1, double m2, double d);	
public:
	vec2d_t Force;
	vec2d_t Velocity;
	vec2d_t Position;

	SimBody(double px = 0.0, double py = 0.0, double vx = 0.0, double vy = 0.0, double mass = 0.0);

	void Tick(double dt);	
	
	void AddForce(const SimBody& b);
	void ResetForce();
	
	double GetMass() const;
};

#endif //BODY_H
