#pragma once
#include "IFeature.h"

class Repeater : virtual public IFeature {
public:
    double Get() override {
        return value;
    }
};