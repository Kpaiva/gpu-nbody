//Coded by Clinton Bale
//02/06/2013

#include "simbody.h"
#include <cassert>
#include <complex>

#ifdef USE_SSE
#include <xmmintrin.h>	
#if !IS_LINUX
#include <intrin.h>
#endif
#endif

SimBody::SimBody(double px /*= 0.0*/, double py /*= 0.0*/, double vx /*= 0.0*/, double vy /*= 0.0*/, double mass /*= 0.0*/) :
	mass_(mass), Velocity(vx,vy), Position(px,py), Force() {
}

inline double SimBody::ComputeGC(double m1, double m2, double d) {
	static const double G = 6.67384 * pow(10.0, -11.0);	
	return (G*m1*m2)/(d*d);
}

void SimBody::Tick(double dt) {
	Velocity.x += dt * (Force.x / mass_);
	Velocity.y += dt * (Force.y / mass_);
	Position.x += dt * Velocity.x;
	Position.y += dt * Velocity.y;
}	

void SimBody::AddForce(const SimBody& b) {
	double dx = b.Position.x - Position.x;
	double dy = b.Position.y - Position.y;
	double r = sqrt(dx*dx + dy*dy);
	double F = ComputeGC(mass_, b.GetMass(), r);
	Force.x += F * (dx / r);
	Force.y += F * (dy / r);
}

void SimBody::ResetForce() {
	Force.x = Force.y = 0;
}

double SimBody::GetMass() const {
	return mass_;
}