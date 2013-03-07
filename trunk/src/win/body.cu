#pragma once
#include <SFML/Graphics.hpp>
#include "../common.h"
#include <cassert>
#include "../sim/simbody.cu"

template <typename T>
class _Body : public _SimBody<T>
{
    sf::Sprite  sprite_;
public:
    _Body(sf::Image *image, T rx = 0.0, T ry = 0.0, T vx = 0.0, T vy = 0.0, T mass = 0.0)
        : sprite_(*image), SimBody(rx, ry, vx, vy, mass)
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