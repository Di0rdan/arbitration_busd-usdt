#pragma once
#include "repeater.h"
#include "data_choosers.h"

class BidPrice0 : public Repeater, public BidPrice0_ {};
class BidPrice1 : public Repeater, public BidPrice1_ {};
class AskPrice0 : public Repeater, public AskPrice0_ {};
class AskPrice1 : public Repeater, public AskPrice1_ {};
