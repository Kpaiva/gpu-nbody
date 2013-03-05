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
#ifdef USE_SSE
	__declspec(align(16)) static const __m128d ssG = _mm_set1_pd(G);		
	__declspec(align(16)) double result;
	
	__m128d ssm1 = _mm_set1_pd(m1);
	__m128d ssm2 = _mm_set1_pd(m2);
	__m128d ssD = _mm_set1_pd(d);

	//result = (ssG*ssm1*ssm2) / (ssD*ssD)
	_mm_store_sd(&result, _mm_div_pd(_mm_mul_pd(ssG, _mm_mul_pd(ssm1, ssm2)), _mm_mul_pd(ssD, ssD)));

	return result;
#else
	return (G*m1*m2)/(d*d);
#endif // USE_SSE
}

void Body::Tick(double dt) {
#ifdef USE_SSE
	__m128d mmDT = _mm_set1_pd(dt);
	__m128d mmMass = _mm_set1_pd(mass_);

	//Velocity.x += dt * (Force.x / mass_);
	_mm_storel_pd(&Velocity.x, _mm_add_pd(_mm_set1_pd(Velocity.x), _mm_mul_pd(mmDT, _mm_div_pd(_mm_set1_pd(Force.x), mmMass))));
	//Velocity.y += dt * (Force.y / mass_);
	_mm_storel_pd(&Velocity.y, _mm_add_pd(_mm_set1_pd(Velocity.y), _mm_mul_pd(mmDT, _mm_div_pd(_mm_set1_pd(Force.y), mmMass))));
		
	//Position.x += dt * Velocity.x;
	_mm_storel_pd(&Position.x, _mm_add_pd(_mm_set1_pd(Position.x), _mm_mul_pd(mmDT, _mm_set1_pd(Velocity.x))));
	//Position.y += dt * Velocity.y;
	_mm_storel_pd(&Position.y, _mm_add_pd(_mm_set1_pd(Position.y), _mm_mul_pd(mmDT, _mm_set1_pd(Velocity.y))));	
#else
	Velocity.x += dt * (Force.x / mass_);
	Velocity.y += dt * (Force.y / mass_);
	Position.x += dt * Velocity.x;
	Position.y += dt * Velocity.y;
#endif //USE_SSE
}	

void Body::AddForce(const Body& b) {
#ifdef USE_SSE
	__declspec(align(16)) double r;

	//dx = b.Position.x - Position.x;
	__m128d dx = _mm_set1_pd(b.Position.x);
	dx = _mm_sub_pd(dx, _mm_set1_pd(Position.x));

	//dy = b.Position.y - Position.y
	__m128d dy = _mm_set1_pd(b.Position.y);
	dy = _mm_sub_pd(dy, _mm_set1_pd(Position.y));

	//mmr = sqrt(dx*dx + dy*dy)
	__m128d mmr = _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(dx,dx),_mm_mul_pd(dy,dy)));
	//Store r to use in the function call below.
	_mm_store_sd(&r, mmr);

	__m128d F = _mm_set1_pd(ComputeGC(mass_, b.GetMass(), r));

	//Force.x = Force.x + (F * (dx / mmr));
	_mm_storel_pd(&Force.x, _mm_add_pd(_mm_set1_pd(Force.x), _mm_mul_pd(F, _mm_div_pd(dx, mmr))));
	//Force.y = Force.y + (F * (dy / mmr));
	_mm_storel_pd(&Force.y, _mm_add_pd(_mm_set1_pd(Force.y), _mm_mul_pd(F, _mm_div_pd(dy, mmr))));		
#else
	double dx = b.Position.x - Position.x;
	double dy = b.Position.y - Position.y;
	double r = sqrt(dx*dx + dy*dy);
	double F = ComputeGC(mass_, b.GetMass(), r);
	Force.x += F * (dx / r);
	Force.y += F * (dy / r);
#endif // USE_SSE
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
