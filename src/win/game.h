//Coded by Clinton Bale
//02/06/2013

#pragma once
#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics/RenderWindow.hpp>
#include "../common.h"

#define FRAMERATE 120

class Game {
private:
	Game(void);
	~Game(void);

	sf::RenderWindow app_;
	sf::Vector2f center_;
	bool running_;
	bool fullscreen_;
	bool showFps_;
	bool showTimer_;
public:
	enum Size {
		WIDTH = 960,
		HEIGHT = 576
	};

	static Game& GetInstance();
	
	sf::RenderWindow& GetApp();
	bool GetFullscreen() const;
	void SetFullscreen(bool full);

	void DrawDebug(void);
	void Init(void);
	int Run(const char* argv);	
};

#endif //GAME_H