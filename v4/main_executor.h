#include <iostream>
#include "models/mm_simple.h"
#include "exchange.h"

template<class Model>
class MainExecutor {
public:
    Exchange exchange;
    Model model;

    std::ofstream data_log;

    uint64_t step_count = 0;
    int64_t data_log_freq;

    void print_data_row(std::ostream& out) {
        DataRow data_row = exchange.GetData();
        out <<
            data_row.timestamp << 
            "," << exchange.Balance0() << 
            "," << exchange.Balance1() << 
            "," << exchange.Converted0() << 
            "," << exchange.Converted1() << 
            "," << exchange.Volume0() << 
            "," << exchange.Volume1() << 
            "," << data_row.bid_pr_0 << 
            "," << data_row.ask_pr_0 << 
            "," << data_row.bid_pr_1 << 
            "," << data_row.ask_pr_1 << 
            "," << data_row.ratio_01 <<
            "," << data_row.ratio_10 <<
            "\n";

    }
    
public:
    MainExecutor(
        Model model_,
        int64_t delay, 
        double comission,
        int64_t orders_freq,
        int64_t limit_wait,
        const std::string& trades_path,
        const std::string& tickers_path,
        const std::string& data_log_path,
        int64_t data_log_freq
    ) : 
        model(model_),
        exchange(
            delay, 
            comission,
            orders_freq,
            limit_wait,
            trades_path,
            tickers_path 
        ),
        data_log(data_log_path),
        data_log_freq(data_log_freq)
    {
        model.SetExchange(&exchange);
        data_log << 
            "timestamp"
            ",balance0"
            ",balance1"
            ",converted0"
            ",converted1"
            ",volume0"
            ",volume1"
            ",bid_pr_0"
            ",ask_pr_0"
            ",bid_pr_1"
            ",ask_pr_1"
            ",ratio_01"
            ",ratio_10"
            "\n";
        // std::cout << "alignof: " << alignof(exchange) << "\n";
    }

    void OnTimer(int64_t timestamp) {
        // std::cout << "MainExecutor::OnTimer(" << timestamp << ")" << std::endl;
        // std::cout << "Exchange::OnTimer(" << timestamp << ")" << std::endl;
        std::optional<DataRow> data_row = exchange.OnTimer(timestamp);
        if (data_row.has_value()) {
            // std::cout << "Model::Update" << std::endl;
            model.Update(data_row.value());
        }
        // std::cout << "Exchange::RunSignals()" << std::endl;
        exchange.RunSignals();

        ++step_count;
        if (step_count % data_log_freq == 0) {
            print_data_row(data_log);
        }
    }

    void PrintStat() const {
        std::cout << "balance0:   " << exchange.Balance0() << "\n";
        std::cout << "balance1:   " << exchange.Balance1() << "\n";
        std::cout << "converted0: " << exchange.Converted0() << "\n";
        std::cout << "converted1: " << exchange.Converted1() << "\n";
        std::cout << "volume0:    " << exchange.Volume0() << "\n";
        std::cout << "volume1:    " << exchange.Volume1() << "\n";
    }
};