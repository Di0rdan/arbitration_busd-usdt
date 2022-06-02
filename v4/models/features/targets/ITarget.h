#pragma once
#include "../../../data_row.h"
#include <exception>

class ITarget {
protected:
    struct Order {
        int64_t timestamp;
        double value;
    };
    std::deque<Order> res;
    
public:
    double delay = 0;
    int tick_delay_count = 0;

    virtual void Update(const DataRow& data_row) {
        throw std::logic_error("ITarget::Update is not implemented\n");
    }
    virtual double Get() {
        throw std::logic_error("ITarget::GetCur is not implemented\n");
    }
    virtual void SetParameter(const std::string& name, double value) {
        if (name=="delay") {
            delay = value;
        } else if (name=="tick_delay_count") {
            tick_delay_count = value;
        } else {
            std::cerr << "IFeature::SetParameter:\n\tunknown parameter " << name << "\n";
        }
    }
};