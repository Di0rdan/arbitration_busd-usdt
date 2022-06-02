#pragma once
#include "averager.h"
#include "data_choosers.h"

class MeanBidPrice0 : public Averager, public BidPrice0_ {};
class MeanBidPrice1 : public Averager, public BidPrice1_ {};
class MeanAskPrice0 : public Averager, public AskPrice0_ {};
class MeanAskPrice1 : public Averager, public AskPrice1_ {};
