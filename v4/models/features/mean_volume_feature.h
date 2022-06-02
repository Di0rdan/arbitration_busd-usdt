#pragma once
#include "averager.h"
#include "data_choosers.h"

class MeanBidVolume0 : public Averager, public BidVolume0_ {};
class MeanBidVolume1 : public Averager, public BidVolume1_ {};
class MeanAskVolume0 : public Averager, public AskVolume0_ {};
class MeanAskVolume1 : public Averager, public AskVolume1_ {};
