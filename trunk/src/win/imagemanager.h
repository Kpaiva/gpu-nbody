//Team Cosmosis

#pragma once
#ifndef IMGMANAGER_H
#define IMGMANAGER_H

#include <SFML/System/NonCopyable.hpp>
#include <SFML/Graphics/Image.hpp>
#include <map>

typedef std::map<std::string, sf::Image*>::iterator asset_it;
#undef LoadImage

class ImageManager : sf::NonCopyable {
private:
	std::map<std::string, sf::Image*> assets_;

	ImageManager(void);
	~ImageManager();

	bool ContainsAsset(const std::string&, asset_it* = (asset_it*)0);
public:
	static ImageManager& GetInstance(void);

	sf::Image* GetImage(const std::string& imageName);
	sf::Image* LoadImage(const std::string& imagePath, const std::string& assetName);
};

#endif //NBODYCONFIG_H