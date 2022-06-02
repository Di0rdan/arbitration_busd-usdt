#include <iostream>
#include <Eigen/Dense>

#include "main_executor.h"

#include "models/mm_simple.h"
#include "models/ml_simple.h"
#include "models/feature_printer.h"
// #include "models/I_ML_Model.h"
#include "models/limit_price_predict.h"

#include "solvers/parameters_parser.h"
#include <iomanip>
#include <string>
#include <fstream>

#ifndef Model
#define Model MM_Simple
#endif

#define DEF2STR(x) #x
#define STR(x) DEF2STR(x)


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "enter the path of the settings file as the first argument\n";
        return 1;
    }

    std::string model_settings_path = "settings/" STR(Model) ".settings.txt";
    
    int delay = 10000;
    int orders_freq = 40000;
    int limit_wait = 3e6;
    double comission = 0.00036;
    std::string trades_path  = "../../research/trades_coursework.csv";
    std::string tickers_path = "../../research/tickers_coursework.csv";
    std::string data_log_path = "data_log.csv";
    int data_log_freq = 1000;

    int freq = 1000;
    int64_t start_timestamp = 1649797200006000ll;
    int64_t finish_timestamp = 1649980799993000ll;
    uint64_t max_step_count = -1ll;

    SET_VARIABLES(argv[1]
        ,model_settings_path
        ,delay
        ,orders_freq
        ,limit_wait
        ,comission
        ,max_step_count
        ,trades_path
        ,tickers_path
        ,data_log_path
        ,data_log_freq
        ,freq
        ,start_timestamp
        ,finish_timestamp
    );

    MainExecutor<Model> executor(
        model_settings_path,
        delay, 
        comission,
        orders_freq,
        limit_wait,
        trades_path,
        tickers_path,
        data_log_path,
        data_log_freq
    );

    std::cerr << "execution started\n";
    std::cerr << std::fixed << std::setprecision(10);
    
    
    int64_t timestamp = start_timestamp, i = 0;
    
    for (; 
        timestamp <= finish_timestamp; 
        timestamp += freq, ++i) {
        if (i > max_step_count) {
            break;
        }
        executor.OnTimer(timestamp);
    }
    std::cerr << "execution finished\n";
    std::cerr << "step count: " << i << "\n";
    std::cerr << "timestamp: " << timestamp << "\n";

    executor.PrintStat();
}
