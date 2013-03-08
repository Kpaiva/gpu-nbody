//Team Cosmosis

#include "imagemanager.h"

ImageManager& ImageManager::GetInstance(void) {
	static ImageManager imageManager;
	return imageManager;
}

ImageManager::ImageManager( void ) {
}

ImageManager::~ImageManager() {
	for(asset_it i = assets_.begin(); i != assets_.end(); i++) {
		if((*i).second) {
			delete (*i).second;
			(*i).second = (sf::Image*)0;
		}
	}
	assets_.clear();
}

bool ImageManager::ContainsAsset( const std::string& assetName, asset_it* tempAsset) {
	if(tempAsset) {
		*tempAsset = assets_.find(assetName);
		if(*tempAsset == assets_.end())
			return false;
		return true;
	}
	else {
		asset_it asset = assets_.find(assetName);
		if(asset == assets_.end())
			return false;
		return true;
	}
}

sf::Image* ImageManager::GetImage( const std::string& assetName ) {
	asset_it asset;
	if(ContainsAsset(assetName, &asset))
		return asset->second;
	return (sf::Image*)NULL;
}

sf::Image* ImageManager::LoadImage( const std::string& imagePath, const std::string& assetName ) {
	if(ContainsAsset(assetName))
		return assets_[assetName];	

	sf::Image* image = new sf::Image;
	if(image->LoadFromFile(imagePath)) {
		assets_.insert(std::make_pair(assetName, image));
		return image;
	}
	else {
		delete image;
	}

	return (sf::Image*)NULL;
}