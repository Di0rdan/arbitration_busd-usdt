#pragma once
#include "../../data_row.h"
#include "../../solvers/ema.h"
#include <exception>
#include <cmath>

class IFeature {
private:
    EMA mean, mean2;
protected:
    double value;
public:
    void SetNorm(double alpha) {
        mean.SetAlpha(alpha);
        mean.SetValue(0.0);
        mean2.SetAlpha(alpha);
        mean2.SetValue(0.0);
    }
    virtual void Update(const DataRow& data_row) {
        throw std::logic_error("IFeature::Update is not implemented\n");
    }
    virtual double Get() {
        throw std::logic_error("IFeature::Update is not implemented\n");
    }
    virtual void SetParameter(const std::string& name, double value) {
        if (name=="norm_alpha") {
            SetNorm(value);
        } else {
            std::cerr << "IFeature::SetParameter:\n\tunknown parameter " << name << "\n";
        }
    }

    double GetNormed() {
        double cur = Get();
        mean.Update(cur);
        mean2.Update(cur * cur);
        double mean_val = mean.Get();
        double mean2_val = mean2.Get();
        if (!isnan((cur - mean_val) / std::sqrt(mean2_val - mean_val * mean_val))) {
            return (cur - mean_val) / std::sqrt(mean2_val - mean_val * mean_val);
        } else {
            std::cerr << "GetNormed -> NaN\n";
            std::cerr << "\tcur = " << cur << "\n";
            std::cerr << "\tmean = " << mean_val << "\n";
            std::cerr << "\tmean2 = " << mean2_val << "\n";
            return 0;
        }
    }
};