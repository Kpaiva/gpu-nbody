//Team Cosmosis

#pragma once
#ifndef BODY_H
#define BODY_H

#include "../common.h"
#include <complex>
#include <cuda_runtime.h>

template <typename T>
class _SimBody {
private:
	/*
	CUDA_CALLABLE_MEMBER T ComputeGC(T m1, T m2, T d) {
		const T G = 6.67384f * pow(10.0f, -11.0f);
		return (G*m1*m2)/(d*d);
	}
	*/
public:
	T Mass;
	vec2_t Force;
	vec2_t Velocity;
	vec2_t Position;

	_SimBody(T px = 0.0f, T py = 0.0f, T vx = 0.0f, T vy = 0.0f, T mass = 0.0f) :
		Mass(mass), Velocity(vx,vy), Position(px,py), Force() { }

	/*	
	CUDA_CALLABLE_MEMBER void Tick(T dt) {
		Velocity.x += dt * (Force.x / Mass);
		Velocity.y += dt * (Force.y / Mass);
		Position.x += dt * Velocity.x;
		Position.y += dt * Velocity.y;
	}

	CUDA_CALLABLE_MEMBER void AddForce(const SimBody& b) {
		T dx = b.Position.x - Position.x;
		T dy = b.Position.y - Position.y;
		T r = sqrt(dx*dx + dy*dy);
		T F = ComputeGC(Mass, b.Mass, r);
		Force.x += F * (dx / r);
		Force.y += F * (dy / r);
	}

	CUDA_CALLABLE_MEMBER void ResetForce() {
		Force = vec2_t();
	}
	*/
};

#endif //BODY_H
