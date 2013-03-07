//Coded by Clinton Bale
//02/06/2013

#include "bodymanager.h"
#include "body.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>

BodyManager& BodyManager::GetInstance() {
	static BodyManager self;
	return self;
}

BodyManager::BodyManager(void) :
 imageManager_(ImageManager::GetInstance()) {
	zoomLevel_ = 128;
	srand((unsigned int)time(NULL));
}

BodyManager::~BodyManager(void) {
	bodies_.clear();
}

void BodyManager::Tick(double gameTime) {
	size_t i = 0;
	size_t j = 0;

	for(i = 0; i < bodies_.size(); ++i) {
		bodies_[i].ResetForce();
		for(j = 0; j < bodies_.size(); ++j) {
			if(i != j) bodies_[i].AddForce(bodies_[j]);			
		}
	}

	for(i = 0; i < bodies_.size(); ++i) {		
		bodies_[i].Tick(gameTime);		
	}
}

bool BodyManager::Init( int count, double radius, sf::RenderWindow* app) {	
	if(app == NULL || count <= 0 || radius <= 0) return false;

	bodies_.clear();
	bodies_.reserve(count*16);
	solarRadius_ = radius;
	app_ = app;

	return true;
}

bool BodyManager::InitFromFile(const char* fileName, sf::RenderWindow* app) {
	char fileStr[260];
	size_t count = 0;
	double radius = 0.0, rx, ry, vx, vy, m;
	
	//Bad parameters
	if(fileName == NULL || app == NULL) return false;

	//Set the render window
	app_ = app;
	//Remove previous loads
	bodies_.clear();

	FILE* file = fopen(fileName, "r");
	
	//File doesn't exist
	if(file == NULL) return false;

	//Read the count of planets and the radius of the universe.
	fscanf(file, "%d\n", &count);
	fscanf(file, "%lf\n", &radius);

	//No count or radius is bad.
	if(count <= 0 || radius <= 0.0) {		
		fclose(file);
		return false;
	}

	//Set the solar radius
	solarRadius_ = radius;
	//Reserve count amount of items, for faster adding.
	bodies_.reserve(count);



	//Add the bodies
	for(size_t i = 0; i < count; ++i) {		
		fscanf(file, "%lf %lf %lf %lf %lf %s\n", &rx, &ry, &vx, &vy, &m, &fileStr);		
		AddBody(Body(imageManager_.GetImage(fileStr), rx, ry, vx, vy, m));
	}

	return !fclose(file);
}

void BodyManager::AddBody(const _Body& body) {
	bodies_.push_back(body);
}

void BodyManager::Render( void ) {
	size_t size = bodies_.size();
	for(size_t i = 0; i < size; ++i) {
		_Body& body = bodies_[i];
		body.SetSpritePosition(body.Position.x*zoomLevel_/solarRadius_, body.Position.y*zoomLevel_/solarRadius_);
		app_->Draw(body);		
	}
}

void BodyManager::ClearBodies() {
	bodies_.clear();
}

double BodyManager::GetSolarRadius() const {
	return solarRadius_;
}

void BodyManager::IncreaseZoom() {
	zoomLevel_ <<= 1;
}

void BodyManager::DecreaseZoom() {
	zoomLevel_ >>= 1;
	if(zoomLevel_ == 0)
		zoomLevel_ = 1;
}

int BodyManager::ZoomLevel() const {
	return zoomLevel_;
}


