//Team Cosmosis

#pragma once
#include <SFML/Graphics.hpp>
#include "../common.h"
#include <cassert>
#include "../sim/simbody.cu"

class _Body 
{
    sf::Sprite  sprite_;
public:
    _Body(sf::Image *image): sprite_(*image)
    {
        sprite_.SetCenter(image->GetWidth() / 2.0f, image->GetHeight() / 2.0f);
    }

    sf::Sprite &Sprite() const
    {
        return (sf::Sprite &)sprite_;
    }

    operator sf::Sprite &() const
    {
        return (sf::Sprite &)sprite_;
    }

    void SetSpritePosition( float x, float y )
    {
        sprite_.SetPosition(x, y);
    }
};