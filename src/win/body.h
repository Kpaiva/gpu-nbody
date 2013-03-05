//Coded by Clinton Bale
//01/14/2013

//Coded by Clinton Bale
//02/06/2013

#pragma once
#ifndef BODY_H
#define BODY_H

#include <SFML/Graphics.hpp>
#include "common.h"

class Body {
private:
	sf::Sprite	sprite_;
	double		mass_;

	inline double ComputeGC(double m1, double m2, double d);	
public:
	sf::Vector2<double> Force;
	sf::Vector2<double> Velocity;
	sf::Vector2<double> Position;

	Body(sf::Image* image, double rx = 0.0, double ry = 0.0, double vx = 0.0, double vy = 0.0, double mass = 0.0);

	void Tick(double dt);	
	
	void AddForce(const Body& b);
	void ResetForce();

	operator sf::Sprite&() const;
	sf::Sprite& Sprite() const;
	void SetSpritePosition(float x, float y);

	double SetMass(double mass);
	double GetMass() const;
};

#endif //BODY_H