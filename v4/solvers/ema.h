#pragma once
#include <cmath>

class EMA {
private:
    double value = NAN;
    double alpha;

public:
    EMA() {}
    EMA(double alpha, double value = 0) : alpha(alpha), value(value) {
    }

    void SetAlpha(double new_alpha) {
        alpha = new_alpha;
    }
    void SetValue(double new_value) {
        value = new_value;
    }

    void Update(double update) {
        if (isnan(value)) {
            value = update;
        }
        value = value * (1 - alpha) + update * alpha;
    }

    double Get() const {
        return value;
    }
};