#pragma once
#include "../exchange.h"
#include <exception>

class IModel {
public:
    Exchange* exchange = nullptr;

public:
    void SetExchange(Exchange* exchange_ptr) {
        exchange = exchange_ptr;
    }

    void Update(const DataRow&);
};