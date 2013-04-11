//Team Cosmosis

#include "common.h"

#if IS_LINUX || IS_SIMULATION

#include "sim/simulation.h"

#if IS_TESTING
#include "sim/simtester.h"
#endif

int main(int argc, char* argv[]) {
#if IS_TESTING
	SimFullTest(10);
#else
	Simulation& simulation = Simulation::GetInstance();
	if(!simulation.Setup(argc, argv))
		return simulation.Run();
#endif
	return 1;
}

#else //WINDOWS GFX

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "win/game.h"

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR str, int) {
	Game& game = Game::GetInstance();
	game.Init();
	return game.Run(str);
}
#endif
