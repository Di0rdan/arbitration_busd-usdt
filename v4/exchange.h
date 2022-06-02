#pragma once
#include <memory> 
#include <set>
#include <deque>
#include <functional>
#include <vector>
#include "data_manager.h"
#include "signal.h"

class Exchange {
private:
    DataManager data_manager;

    const int64_t delay;
    const int64_t limit_wait;
    const double comission;
    const int64_t max_limits_count = 5000;

    struct MarketOrder {
        double volume;
        int64_t timestamp;

        MarketOrder(double volume, int64_t timestamp) : volume(volume), timestamp(timestamp) {}
        MarketOrder(const MarketOrder& other) : volume(other.volume), timestamp(other.timestamp) {
        }
    };
    struct LimitOrder {
        double volume;
        double price;
        int64_t timestamp;

        LimitOrder(double volume, double price, int64_t timestamp) : volume(volume), price(price), timestamp(timestamp) {}

        bool operator< (const LimitOrder& other) const {
            return price > other.price;
        }
    };

    std::deque<MarketOrder> market0, market1;
    std::deque<LimitOrder> limit0, limit1;
    std::multiset<LimitOrder> limit_orders0, limit_orders1;

    DataRow data_row;

    double balance0 = 0;
    double balance1 = 0;

    double volume0 = 0;
    double volume1 = 0;

    int64_t last_order_timestamp = -1;

    const int64_t orders_freq;

    void update_limits_0() {
        return;
        if (limit_orders0.size() < max_limits_count) {
            return;
        }
        std::vector<LimitOrder> limit0_list;
        limit0_list.reserve(max_limits_count);
        
        for (auto& order : limit_orders0) {
            if (order.timestamp + limit_wait >= data_row.timestamp) {
                limit0_list.push_back(order);
            }
        }

        limit_orders0 = std::multiset<LimitOrder>(
            limit0_list.begin(), limit0_list.end());
    }

    void update_limits_1() {
        return;
        if (limit_orders1.size() < max_limits_count) {
            return;
        }
        std::vector<LimitOrder> limit1_list;
        limit1_list.reserve(max_limits_count);
        
        for (auto& order : limit_orders1) {
            if (order.timestamp + limit_wait >= data_row.timestamp) {
                limit1_list.push_back(order);
            }
        }

        limit_orders1 = std::multiset<LimitOrder>(
            limit1_list.begin(), limit1_list.end());
    }

public:
    Exchange(
        int64_t delay, 
        double comission,
        int64_t orders_freq,
        int64_t limit_wait,
        const std::string& trades_path,
        const std::string& tickers_path 
    ) :
        delay(delay),
        comission(comission),
        orders_freq(orders_freq),
        limit_wait(limit_wait),
        data_manager(trades_path, tickers_path) 
    {}

    std::optional<DataRow> OnTimer(int64_t timestamp) {
        std::optional<DataRow> cur_row = data_manager.GetData(timestamp);
        if (cur_row.has_value()) {
            data_row = cur_row.value();
        }
        // std::cout << "market0.size() = " << market0.size() << std::endl;
        return cur_row;
    }

    void MarketMarketOrder_0(double volume) {
        // std::cout << "Exchange::MarketMarketOrder_0(" << volume << ")" << std::endl;
        // std::cout << "pushing to market0, cur size = " << market0.size() << std::endl;
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            market0.emplace_back(volume, data_row.timestamp);
            last_order_timestamp = data_row.timestamp;
        }
        // std::cout << "pushed order to market0" << std::endl;
    } 
    void MarketMarketOrder_1(double volume) {
        // std::cout << "Exchange::MarketMarketOrder_1(" << volume << ")" << std::endl;
        // std::cout << "pushing to market1, cur size = " << market1.size() << std::endl;
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            market1.emplace_back(volume, data_row.timestamp);
            last_order_timestamp = data_row.timestamp;
        }
        // std::cout << "pushed order to market1" << std::endl;
    } 

    void LimitMarketOrder_0(double volume, double price) {
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            limit0.emplace_back(volume, price, data_row.timestamp);
            update_limits_0();
            last_order_timestamp = data_row.timestamp;
        }
    }
    void LimitMarketOrder_1(double volume, double price) {
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            limit1.emplace_back(volume, price, data_row.timestamp);
            update_limits_1();
            last_order_timestamp = data_row.timestamp;
        }
    }
    void LimitMarketOrder_0(double volume) {
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            limit0.emplace_back(volume, data_row.bid_pr_0, data_row.timestamp);
            update_limits_0();
            last_order_timestamp = data_row.timestamp;
        }
    }
    void LimitMarketOrder_1(double volume) {
        if (last_order_timestamp + orders_freq <= data_row.timestamp) {
            limit1.emplace_back(volume, data_row.bid_pr_1, data_row.timestamp);
            update_limits_1();
            last_order_timestamp = data_row.timestamp;
        }
    }
    
    void RunSignals() {
        while (!market0.empty() && market0.front().timestamp + delay <= data_row.timestamp) {
            balance0 -= market0.front().volume;
            volume0 += market0.front().volume;
            balance1 += market0.front().volume * data_row.bid_pr_1 / data_row.ask_pr_0 * (1 - comission);
            market0.pop_front();
        }
        while (!market1.empty() && market1.front().timestamp + delay <= data_row.timestamp) {
            balance1 -= market1.front().volume;
            volume1 += market1.front().volume;
            balance0 += market1.front().volume * data_row.bid_pr_0 / data_row.ask_pr_1 * (1 - comission);
            market1.pop_front();
        }

        while (!limit0.empty() && limit0.front().timestamp + delay <= data_row.timestamp) {
            limit_orders0.insert(limit0.front());
            limit0.pop_front();
        }
        while (!limit1.empty() && limit1.front().timestamp + delay <= data_row.timestamp) {
            limit_orders1.insert(limit1.front());
            limit1.pop_front();
        }
        
        while (!limit_orders0.empty() && limit_orders0.begin()->price > data_row.bid_pr_0) {
            if (limit_orders0.begin()->timestamp + delay + limit_wait > data_row.timestamp) {
                balance0 -= limit_orders0.begin()->volume;
                volume0 += limit_orders0.begin()->volume;
                balance1 += limit_orders0.begin()->volume * data_row.bid_pr_1 / limit_orders0.begin()->price * (1 - comission / 2);
            }
            limit_orders0.erase(limit_orders0.begin());
        }
        while (!limit_orders1.empty() && limit_orders1.begin()->price > data_row.bid_pr_1) {
            if (limit_orders1.begin()->timestamp + delay + limit_wait > data_row.timestamp) {
                balance1 -= limit_orders1.begin()->volume;
                volume1 += limit_orders1.begin()->volume;
                balance0 += limit_orders1.begin()->volume * data_row.bid_pr_0 / limit_orders1.begin()->price * (1 - comission / 2);
            }
            limit_orders1.erase(limit_orders1.begin());
        }
    }

    double Balance0() const {
        return balance0;
    }
    double Balance1() const {
        return balance1;
    }

    double Converted0() const {
        double converted0 = balance0;
        if (balance1 > 0) {
            converted0 += balance1 * data_row.bid_pr_0 / data_row.ask_pr_1 * (1 - comission);
        } else {
            converted0 += balance1 / data_row.bid_pr_1 * data_row.ask_pr_0 / (1 - comission);
        }
        return converted0;
    }

    double Converted1() const {
        double converted1 = balance1;
        if (balance0 > 0) {
            converted1 += balance0 * data_row.bid_pr_1 / data_row.ask_pr_0 * (1 - comission);
        } else {
            converted1 += balance0 / data_row.bid_pr_0 * data_row.ask_pr_1 / (1 - comission);
        }
        return converted1;
    }

    double Volume0() const {
        return volume0;
    }

    double Volume1() const {
        return volume1;
    }

    DataRow GetData() const {
        return data_row;
    }
};