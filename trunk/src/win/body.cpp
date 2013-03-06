//Coded by Clinton Bale
//02/06/2013

#include "body.h"
#include <xmmintrin.h>
#include <cassert>

Body::Body(sf::Image* image, double rx /*= 0.0*/, double ry /*= 0.0*/, double vx /*= 0.0*/, double vy /*= 0.0*/, double mass /*= 0.0*/) 
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

inline double Body::ComputeGC(double m1, double m2, double d) {
	static const double G = 6.67384 * pow(10.0, -11.0);	
	return (G*m1*m2)/(d*d);
}

void Body::Tick(double dt) {
	Velocity.x += dt * (Force.x / mass_);
	Velocity.y += dt * (Force.y / mass_);
	Position.x += dt * Velocity.x;
	Position.y += dt * Velocity.y;
}	

void Body::AddForce(const Body& b) {
	double dx = b.Position.x - Position.x;
	double dy = b.Position.y - Position.y;
	double r = sqrt(dx*dx + dy*dy);
	double F = ComputeGC(mass_, b.GetMass(), r);
	Force.x += F * (dx / r);
	Force.y += F * (dy / r);
}

void Body::ResetForce() {
	Force.x = Force.y = 0;
}

double Body::SetMass(double mass) {
	return mass_ = mass;
}

double Body::GetMass() const {
	return mass_;
}

sf::Sprite& Body::Sprite() const {
	return (sf::Sprite&)sprite_;
}

Body::operator sf::Sprite&() const {
	return (sf::Sprite&)sprite_;
}

void Body::SetSpritePosition( float x, float y ) {
	sprite_.SetPosition(x, y);
}
