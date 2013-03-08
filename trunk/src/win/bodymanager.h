//Coded by Clinton Bale
//02/06/2013

#pragma once
#ifndef _BODYMANAGER_H_
#define _BODYMANAGER_H_

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>
#include "imagemanager.h"
#include "body.cu"
#include "../sim/simbody.cu"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "imagemanager.h"

class _Body;

class BodyManager : sf::NonCopyable {
	BodyManager(void);
	~BodyManager(void);

	thrust::device_vector<SimBody> bodies_;
	std::vector<_Body> sprites_;
	ImageManager& imageManager_;
	sf::RenderWindow* app_;
	float solarRadius_;	
	size_t zoomLevel_;

	unsigned numBlocks_;
    unsigned numThreads_;
    BodyArray arr_;
public:
	static BodyManager& GetInstance();
		
	void Tick(float gameTime);
	void Render(void);
	bool Init(int count, float radius, sf::RenderWindow* app);
	bool InitFromFile(const char* fileName, sf::RenderWindow* app);
	void AddBody(const SimBody &body);
	void AddBodySprite(const _Body &body);
	float GetSolarRadius() const;
	void IncreaseZoom();
	void DecreaseZoom();
	int ZoomLevel() const;
	unsigned getBlocks() const;
	unsigned getThreads() const;
};
#endif