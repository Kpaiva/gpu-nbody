#pragma once
#include <SFML/Graphics.hpp>
#include "common.h"
#include <cassert>
#include <../sim/simbody.cu>

template <typename T>
class _Body : _SimBody<T>
{
    sf::Sprite  sprite_;
public:
    Body(sf::Image *image, double rx = 0.0, double ry = 0.0, double vx = 0.0, double vy = 0.0, double mass = 0.0)
        : sprite_(*image), _SimBody(rx, ry, vx, vy, mass)
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

    CUDA_CALLABLE_MEMBER void SetSpritePosition( float x, float y )
    {
        sprite_.SetPosition(x, y);
    }
};