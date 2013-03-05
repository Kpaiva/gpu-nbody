//Coded by Clinton Bale
//02/06/2013

#include "win/game.h"
#include "win/bodymanager.h"
#include "win/body.h"
#include "win/imagemanager.h"
#include "timer.h"

using namespace sf;

Game::Game(void) :
	center_(0.0, 0.0) {
}
Game::~Game(void)
{
	app_.Close();	
}

Game& Game::GetInstance() {
	static Game self;
	return self;
}

void Game::Init() {
	SetFullscreen(false);
	app_.SetFramerateLimit(FRAMERATE);

	ImageManager& manager = ImageManager::GetInstance();
	manager.LoadImage("images/earth.png", "earth.gif");
	manager.LoadImage("images/sun.png", "sun.gif");
	manager.LoadImage("images/saturn.png", "saturn.gif");
	manager.LoadImage("images/uranus.png", "uranus.gif");
	manager.LoadImage("images/venus.png", "venus.gif");
	manager.LoadImage("images/pluto.png", "pluto.gif");
	manager.LoadImage("images/jupiter.png", "jupiter.gif");
	manager.LoadImage("images/mars.png", "mars.gif");
	manager.LoadImage("images/mercury.png", "mercury.gif");
	manager.LoadImage("images/neptune.png", "neptune.gif");
	manager.LoadImage("images/blackhole.png", "blackhole.gif");
	manager.LoadImage("images/death_star.png", "death_star.gif");
	manager.LoadImage("images/endor.png", "endor.gif");
	manager.LoadImage("images/ninjasquirrel_red.png", "ninjasquirrel_red.gif");
	manager.LoadImage("images/rebel_cruiser.png", "rebel_cruiser.gif");
	manager.LoadImage("images/squirrel.png", "squirrel.gif");
	manager.LoadImage("images/star.png", "star.gif");
	manager.LoadImage("images/star_destroyer.png", "star_destroyer.gif");
	manager.LoadImage("images/blackhole.png", "blackhole.gif");
	manager.LoadImage("images/asteroid.png", "asteroid.gif");
	manager.LoadImage("images/acorn3.png", "acorn3.gif");
	manager.LoadImage("images/acorn-1.png", "acorn-1.gif");
	manager.LoadImage("images/squirrel.png", "squirrel.gif");
	manager.LoadImage("images/ninjasquirrel_red.png", "ninjasquirrel_red.gif");
	
#ifdef _DEBUG
	showFps_ = showTimer_ = true;
#else
	showFps_ = showTimer_ = false;
#endif
}

void Game::SetFullscreen(bool full)
{
	if (app_.IsOpened())
		app_.Close();

	int style = full ? sf::Style::Fullscreen : sf::Style::Close;
	sf::WindowSettings settings;
	settings.DepthBits = 24;
	settings.StencilBits = 8;
	settings.AntialiasingLevel = 16;

	app_.Create(sf::VideoMode(Game::WIDTH, Game::HEIGHT, 32), "NBody", style, settings);	
	fullscreen_ = full;

	if (!full) {
		// center window on desktop
		sf::VideoMode desktop = sf::VideoMode::GetDesktopMode();
		app_.SetPosition((desktop.Width - WIDTH) / 2, (desktop.Height - HEIGHT) / 2);
	}
}

void Game::DrawDebug(void)
{
	static char c_fpsStr[8] = "fps : ";
	static char c_timeStr[32] = "time: ";
	static sf::String fpsStr(c_fpsStr);
	static sf::String timerStr(c_fpsStr);
	static Timer timer;
	static sf::Clock clock;	
	if(showFps_) {
		fpsStr.SetPosition(center_.x-Game::WIDTH/2, center_.y-Game::HEIGHT/2);
		fpsStr.SetSize(15);
		if(clock.GetElapsedTime()>=0.5) {
			//Every 0.5 seconds
			sprintf(c_fpsStr, "fps: %d\n", (int)(30.0/clock.GetElapsedTime()));
			fpsStr.SetText(c_fpsStr);
			clock.Reset();
		}
		app_.Draw(fpsStr);
	}
	if(showTimer_) {
		timer.stop();
		timerStr.SetPosition(center_.x-Game::WIDTH/2, center_.y-Game::HEIGHT/2+15);
		timerStr.SetSize(15);
		sprintf(c_timeStr, "time: %lf\n", timer.getElapsedTimeInMilliSec());
		timerStr.SetText(c_timeStr);
		app_.Draw(timerStr);
		timer.start();		
	}	
}

int Game::Run(const char* argv)
{
	running_ = true;	
	
	BodyManager& bm = BodyManager::GetInstance();	
	bm.InitFromFile(argv[0] ? argv : "sbh3.txt", &app_);

	Event event;	
	double timeStep = 25000;

	bool mouseMove = false;
	Vector2i mouseLoc;
	while(running_) {
		while (app_.GetEvent(event)) {
			if(event.Type == Event::KeyPressed) {
				switch(event.Key.Code) {
				case Key::Escape:
					running_ = false;
					break;
				case Key::F1:
					showFps_ = !showFps_;
					showTimer_ = !showTimer_;
					break;
				case Key::F2:
					SetFullscreen(!fullscreen_);
					break;
				case Key::W:
					center_.y -= 10;
					break;
				case Key::S:
					center_.y += 10;
					break;
				case Key::A:
					center_.x -= 10;
					break;
				case Key::D:
					center_.x += 10;
					break;
				case Key::Add:
					timeStep += 10000;
					break;
				case Key::Subtract:
					timeStep -= 10000;
					break;
				}			
			}
			else if(event.Type == Event::MouseWheelMoved) {
				if(event.MouseWheel.Delta >= 1) {
					bm.IncreaseZoom();
				}
				else {
					bm.DecreaseZoom();
				}
			}
			else if(event.Type == Event::MouseButtonReleased && event.MouseButton.Button == sf::Mouse::Left) {
				double x = event.MouseButton.X-app_.GetView().GetHalfSize().x+center_.x;
				double y = event.MouseButton.Y-app_.GetView().GetHalfSize().y+center_.y;
				double r = bm.GetSolarRadius();

				x = x * r / bm.ZoomLevel();
				y = y * r / bm.ZoomLevel();
				bm.AddBody(Body(ImageManager::GetInstance().GetImage("earth.gif"), x, y, 0, 0, 5.974e24));
			}
			else if(event.Type == Event::MouseButtonPressed && event.MouseButton.Button == sf::Mouse::Right) {
				mouseMove = true;
				mouseLoc.x = event.MouseButton.X;
				mouseLoc.y = event.MouseButton.Y;
			}
			else if(event.Type == Event::MouseButtonReleased && event.MouseButton.Button == sf::Mouse::Right) {
				mouseMove = false;
			}
		}		
		app_.Clear();		
		
		//update
		bm.Tick(timeStep);
				
		if(mouseMove) {
			const float moveSpeed = 2.0f;
			float dx = app_.GetInput().GetMouseX() - mouseLoc.x;
			float dy = app_.GetInput().GetMouseY() - mouseLoc.y;
			float length = sqrt(dx*dx + dy*dy);
			if (length != 0)
			{
				Vector2f n(dx / length, dy / length);
				n *= moveSpeed + (length / 10.0f);
				center_.x += n.x;
				center_.y += n.y;
			}
		}		
		//------

		//render
		if(app_.GetView().GetCenter() != center_) {
			app_.SetView(View(center_, Vector2f(Game::WIDTH/2,Game::HEIGHT/2)));
		}	

		bm.Render();	

		DrawDebug();
		//------

		app_.Display();
	}
	return EXIT_SUCCESS;
}

sf::RenderWindow& Game::GetApp() {
	return app_;
}