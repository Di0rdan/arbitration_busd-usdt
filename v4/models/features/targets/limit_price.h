#include "ITarget.h"
#include "../../../solvers/queue_max.h"

class LimitPrice : public ITarget {
protected:
    struct Best {
        int64_t timestamp;
        double price;
        bool operator< (const Best& other) const {
            return price > other.price;
        }
    };

    QueueMax<Best> bests;
public:
    void UpdatePrice(int64_t timestamp, double value) {
        while(bests.Size() && bests.Front().timestamp + delay <= timestamp) {
            bests.Pop();
        }
        bests.Push({timestamp, value});
    }

    double Get() override {
        // return bests.Min().price;
        // return bests.Max().price;
        double min = 1.0/0.0;
        for (const auto& val : bests.queue) {
            min = std::min(min, val.value.price);
        }
        // if (min != bests.Max().price)
        //     std::cerr << "FUCK\n";
        return min;
    }
};

class LimitPriceBid0 : public LimitPrice {
public:
    void Update(const DataRow& data_row) override {
        UpdatePrice(data_row.timestamp, data_row.bid_pr_0);
    }
};

class LimitPriceBid1 : public LimitPrice {
public:
    void Update(const DataRow& data_row) override {
        UpdatePrice(data_row.timestamp, data_row.bid_pr_1);
    }
};