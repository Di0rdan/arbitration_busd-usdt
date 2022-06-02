#pragma once
#include "repeater.h"

class Averager : virtual public Repeater {
protected:
    EMA mean;
public:
    double Get() override {
        mean.Update(Repeater::Get());
        return mean.Get();
    }

    void SetParameter(const std::string& name, double value) override {
        if (name == "mean_alpha") {
            mean.SetAlpha(value);
        } else {
            Repeater::SetParameter(name, value);
        }
    }
};