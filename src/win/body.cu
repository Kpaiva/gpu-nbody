#pragma once
#include <SFML/Graphics.hpp>
#include "common.h"
#include <xmmintrin.h>
#include <cassert>
#include <cuda_runtime.h>

class Body {
	sf::Sprite	sprite_;
	double		mass_;

	inline double ComputeGC(double m1, double m2, double d);	
public:
	sf::Vector2<double> Force;
	sf::Vector2<double> Velocity;
	sf::Vector2<double> Position;

	Body(sf::Image* image, double rx = 0.0, double ry = 0.0, double vx = 0.0, double vy = 0.0, double mass = 0.0)
		: sprite_(*image) {	
		sprite_.SetCenter(image->GetWidth()/2.0f, image->GetHeight()/2.0f);
		mass_ = mass;
		Velocity.x = vx;
		Velocity.y = vy;
		Force.x = 0.0;
		Force.y = 0.0;
		Position.x = rx;
		Position.y = ry;
	}

	CUDA_CALLABLE_MEMBER void Tick(double dt) {
		Velocity.x += dt * (Force.x / mass_);
		Velocity.y += dt * (Force.y / mass_);
		Position.x += dt * Velocity.x;
		Position.y += dt * Velocity.y;
	}	
	
	CUDA_CALLABLE_MEMBER void AddForce(const Body& b) {
		double dx = b.Position.x - Position.x;
		double dy = b.Position.y - Position.y;
		double r = sqrt(dx*dx + dy*dy);
		double F = ComputeGC(mass_, b.GetMass(), r);
		Force.x += F * (dx / r);
		Force.y += F * (dy / r);
	}
	
	CUDA_CALLABLE_MEMBER void ResetForce() {
		Force.x = Force.y = 0;
	}

	sf::Sprite& Sprite() const {
		return (sf::Sprite&)sprite_;
	}

	operator sf::Sprite&() const {
		return (sf::Sprite&)sprite_;
	}

	CUDA_CALLABLE_MEMBER void SetSpritePosition( float x, float y ) {
		sprite_.SetPosition(x, y);
	}

	double SetMass(double mass) {
		return mass_ = mass;
	}

	double GetMass() const {
		return mass_;
	}

	//no destructor
};