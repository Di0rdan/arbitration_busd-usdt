#pragma once
#include "../exchange.h"

class IModel {
public:
    Exchange* exchange = nullptr;

public:
    void SetExchange(Exchange* exchange_ptr) {
        exchange = exchange_ptr;
    }
};