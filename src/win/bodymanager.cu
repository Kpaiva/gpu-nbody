#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>
#include "body.cu"
#include <fstream>
#include <cmath>
#include <vector>

#include "imagemanager.h"

void __global__ RenderK(Body* bodies) {
/*
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Body& body = bodies[idx];
	body.SetSpritePosition(body.Position.x*zoomLevel_/solarRadius_, body.Position.y*zoomLevel_/solarRadius_);
*/
}

void __global__ TickTop(Body* bodies) {
/*
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < bodies.size) {
		bodies[idx].ResetForce();

		for(size_t j = 0; j < bodies.size; ++j)
			if(idx != j)
				bodies[idx].AddForce(bodies[j]);
	}
*/
}

void __global__ TickBottom((Body* bodies, float time) {
/*
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < bodies.size) {
		bodies[idx].Tick(time);
*/
}

class Body;

class BodyManager : sf::NonCopyable {
	BodyManager(void) 
		: imageManager_(ImageManager::GetInstance()) {
		zoomLevel_ = 128;
		srand((unsigned int)time(NULL));
	}

	~BodyManager(void) {
		bodies_.clear();
	}

	std::vector<Body> bodies_;
	ImageManager& imageManager_;
	sf::RenderWindow* app_;
	double solarRadius_;	
	size_t zoomLevel_;
public:
	static BodyManager& GetInstance() {
		static BodyManager self;
		return self;
	}

	void BodyManager::Render( void ) {
	/*
		RenderK<<<#,#>>>(bodies_);
		cudaDeviceSynchronize();

		size_t size = bodies_.size();
		for(size_t i = 0; i < size; ++i) {
			//Could put this into a for loop to draw (on the host)
			app_->Draw(body);		
		}
	*/
	}

	bool Init(int count, double radius, sf::RenderWindow* app) {	
		if(app == NULL || count <= 0 || radius <= 0) return false;

		bodies_.clear();
		bodies_.reserve(count*16);
		solarRadius_ = radius;
		app_ = app;

		return true;
	}

	bool InitFromFile(const char* fileName, sf::RenderWindow* app) {
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
		//Make this part into a kernel ?
		for(size_t i = 0; i < count; ++i) {		
			fscanf(file, "%lf %lf %lf %lf %lf %s\n", &rx, &ry, &vx, &vy, &m, &fileStr);		
			AddBody(Body(imageManager_.GetImage(fileStr), rx, ry, vx, vy, m));
		}

		return !fclose(file);
	}

	CUDA_CALLABLE_MEMBER void AddBody(const Body& body) {
		bodies_.push_back(body);
	} 

	void ClearBodies() {
		bodies_.clear();
	}

	double GetSolarRadius() const {
		return solarRadius_;
	}

	void IncreaseZoom() {
		zoomLevel_ <<= 1;
	}

	void DecreaseZoom() {
		zoomLevel_ >>= 1;
		if(zoomLevel_ == 0)
			zoomLevel_ = 1;
	}

	int ZoomLevel() const {
		return zoomLevel_;
	}
};