//Team Cosmosis

#include "common.h"

#if IS_LINUX || IS_SIMULATION

#include "sim/simulation.h"

#if IS_LINUX
int main(int argc, char* argv[]) {
#else

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
	char** argv = __argv;
	int argc = __argc;
	//Setup a console for our windows application.
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
#endif
	Simulation& simulation = Simulation::GetInstance();
	if(!simulation.Setup(argc, argv))
		return simulation.Run();
	return 1;
}

#else //WINDOWS NO SIMULATION

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "win/game.h"

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR str, int) {
	Game& game = Game::GetInstance();
	game.Init();
	return game.Run(str);
}
#endif
