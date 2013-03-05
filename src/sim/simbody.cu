//Team Cosmosis

#pragma once
#ifndef BODY_H
#define BODY_H

#include "../common.h"
#include <complex>
#include <cuda_runtime.h>

#if USE_DOUBLE_PRECISION
class SimBody {
private:
	CUDA_CALLABLE_MEMBER double ComputeGC(double m1, double m2, double d) {
		const double G = 6.67384 * pow(10.0, -11.0);
		return (G*m1*m2)/(d*d);
	}
public:
	double Mass;
	vec2_t Force;
	vec2_t Velocity;
	vec2_t Position;

	SimBody(double px = 0.0, double py = 0.0, double vx = 0.0, double vy = 0.0, double mass = 0.0) :
		Mass(mass), Velocity(vx,vy), Position(px,py), Force() { }

	CUDA_CALLABLE_MEMBER void Tick(double dt) {
		Velocity.x += dt * (Force.x / Mass);
		Velocity.y += dt * (Force.y / Mass);
		Position.x += dt * Velocity.x;
		Position.y += dt * Velocity.y;
	}

	CUDA_CALLABLE_MEMBER void AddForce(const SimBody& b) {
		double dx = b.Position.x - Position.x;
		double dy = b.Position.y - Position.y;
		double r = sqrt(dx*dx + dy*dy);
		double F = ComputeGC(Mass, b.Mass, r);
		Force.x += F * (dx / r);
		Force.y += F * (dy / r);
	}

	CUDA_CALLABLE_MEMBER void ResetForce() {
		Force = vec2_t();
	}
};

#else

class SimBody {
private:
	CUDA_CALLABLE_MEMBER float ComputeGC(float m1, float m2, float d) {
		const float G = 6.67384f * pow(10.0f, -11.0f);
		return (G*m1*m2)/(d*d);
	}
public:
	float Mass;
	vec2_t Force;
	vec2_t Velocity;
	vec2_t Position;

	SimBody(float px = 0.0f, float py = 0.0f, float vx = 0.0f, float vy = 0.0f, float mass = 0.0f) :
		Mass(mass), Velocity(vx,vy), Position(px,py), Force() {	}

	CUDA_CALLABLE_MEMBER void Tick(float dt) {
		Velocity.x += dt * (Force.x / Mass);
		Velocity.y += dt * (Force.y / Mass);
		Position.x += dt * Velocity.x;
		Position.y += dt * Velocity.y;
	}

	CUDA_CALLABLE_MEMBER void AddForce(const SimBody& b) {
		float dx = b.Position.x - Position.x;
		float dy = b.Position.y - Position.y;
		float r = sqrt(dx*dx + dy*dy);
		float F = ComputeGC(Mass, b.Mass, r);
		Force.x += F * (dx / r);
		Force.y += F * (dy / r);
	}

	CUDA_CALLABLE_MEMBER void ResetForce() {
		Force = vec2_t();
	}
};

#endif //USE_DOUBLE_PRECISION

#endif //BODY_H
