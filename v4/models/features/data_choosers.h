#pragma once
#include "IFeature.h"

class BidPrice0_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_pr_0;
    }
};

class AskPrice0_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.ask_pr_0;
    }
};

class BidPrice1_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_pr_1;
    }
};

class AskPrice1_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.ask_pr_1;
    }
};

class BidVolume0_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_vol_0;
    }
};

class AskVolume0_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.ask_vol_0;
    }
};

class BidVolume1_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_vol_1;
    }
};

class AskVolume1_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.ask_vol_1;
    }
};

class Convert0_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_pr_1 / data_row.ask_pr_0;
    }
};

class Convert1_ : virtual public IFeature {
public:
    void Update(const DataRow& data_row) override {
        value = data_row.bid_pr_0 / data_row.ask_pr_1;
    }
};
