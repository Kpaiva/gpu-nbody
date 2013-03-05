//Coded by Clinton Bale
//02/06/2013

#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>

#include "imagemanager.h"

class Body;

class BodyManager : sf::NonCopyable {
	BodyManager(void);
	~BodyManager(void);

	std::vector<Body> bodies_;
	ImageManager& imageManager_;
	sf::RenderWindow* app_;
	double solarRadius_;	
	size_t zoomLevel_;
public:
	static BodyManager& GetInstance();
		
	void Tick(double gameTime);
	void Render(void);
	bool Init(int count, double radius, sf::RenderWindow* app);
	bool InitFromFile(const char* fileName, sf::RenderWindow* app);
	void AddBody(const Body& body);
	void ClearBodies();
	double GetSolarRadius() const;

	void IncreaseZoom();
	void DecreaseZoom();
	int ZoomLevel() const;
};

