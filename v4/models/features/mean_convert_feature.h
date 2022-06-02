#pragma once
#include "averager.h"
#include "data_choosers.h"

class MeanConvert0 : public Averager, public Convert0_ {};
class MeanConvert1 : public Averager, public Convert1_ {};
