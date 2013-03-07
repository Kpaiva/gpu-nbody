#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>
#include "body.cu"
#include <fstream>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "imagemanager.h"
#include "../sim/simbody.cu"
#include "../sim/simulation.cu"

__global__ void TickTop(BodyArray bodies) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < bodies.size) {
		bodies.array[idx].ResetForce();

		for(size_t j = 0; j < bodies.size; ++j)
			if(idx != j)
				bodies.array[idx].AddForce(bodies.array[j]);
	}
}

__global__ void TickBottom(BodyArray bodies, float time) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < bodies.size) 
		bodies.array[idx].Tick(time);
}



class BodyManager : sf::NonCopyable {
	BodyManager(void) 
		: imageManager_(ImageManager::GetInstance()) {
		zoomLevel_ = 128;
	}

	~BodyManager(void) {
		bodies_.clear();
	}

	thrust::device_vector<SimBody> bodies_;
	std::vector<_Body> sprites_;
    BodyArray arr_;
	ImageManager& imageManager_;
	sf::RenderWindow* app_;
	double solarRadius_;	
	size_t zoomLevel_;

    unsigned numBlocks_;
    unsigned numThreads_;

public:
	static BodyManager& GetInstance() {
		static BodyManager self;
		return self;
	}

	void Render( void ) {
		size_t size = bodies_.size();
		for(size_t i = 0; i < size; ++i) {
			SimBody body = bodies_[i];
			sprites_[i].SetSpritePosition(body.Position.x*zoomLevel_/solarRadius_, body.Position.y*zoomLevel_/solarRadius_);
			app_->Draw(sprites_[i]);		
		}
	}

    void Tick(float timeStep) {
        TickTop<<<numBlocks_, numThreads_>>>(arr_);
        if (cudaDeviceSynchronize() != cudaSuccess)
            std::cout << "Error Tick!" << std::endl;
        TickBottom<<<numBlocks_, numThreads_>>>(arr_, timeStep);
        if (cudaDeviceSynchronize() != cudaSuccess)
            std::cout << "Error Tick!" << std::endl;
    }

	bool Init(int count, double radius, sf::RenderWindow* app) {	
		if(app == NULL || count <= 0 || radius <= 0) return false;

		bodies_.clear();
		bodies_.reserve(count*16);
		sprites_.reserve(count*16);
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
		sprites_.reserve(count);

        // ------ kernel launch configurations starts here
        int dev;
        cudaDeviceProp prop;

        if (cudaGetDevice(&dev) != cudaSuccess){
            std::cout << "Error 1" << std::endl;
            return 1;
        }

        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess){
            std::cout << "Error 1" << std::endl;
            return 1;
        }

        numThreads_ = prop.maxThreadsDim[0];
        numBlocks_ = (count + numThreads_ - 1) / numThreads_;

		//Add the bodies
		//Make this part into a kernel ?
		for(size_t i = 0; i < count; ++i) {		
			fscanf(file, "%lf %lf %lf %lf %lf %s\n", &rx, &ry, &vx, &vy, &m, &fileStr);		
			AddBody(SimBody(rx, ry, vx, vy, m));
			AddBodySprite(_Body(imageManager_.GetImage(fileStr)));
		}

        arr_ = MakeArray(bodies_);

		return !fclose(file);
	}

	CUDA_CALLABLE_MEMBER void AddBody(const SimBody& body) {
		bodies_.push_back(body);
	} 

	void AddBodySprite(const _Body& body) {
		sprites_.push_back(body);
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

    unsigned getBlocks() const {
        return numBlocks_;
    }

    unsigned getThreads() const {
        return numThreads_;
    }
};