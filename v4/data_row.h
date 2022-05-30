#pragma once
#include <iostream>

struct DataRow {
    int64_t timestamp = -1;

    double bid_pr_0;
    double ask_pr_0;
    double bid_vol_0;
    double ask_vol_0;
    
    double bid_pr_1;
    double ask_pr_1;
    double bid_vol_1;
    double ask_vol_1;

    double ratio_01;
    double ratio_10;
};

std::ostream& operator<<(std::ostream& out, const DataRow& data_row) {
    out << "timestamp: " << data_row.timestamp << "\n";
    out << "\tbid_pr_0: " << data_row.bid_pr_0 << "\n";
    out << "\task_pr_0: " << data_row.ask_pr_0 << "\n";
    
    out << "\tbid_pr_1: " << data_row.bid_pr_1 << "\n";
    out << "\task_pr_1: " << data_row.ask_pr_1 << "\n";
    
    out << "\tratio 01:    " << data_row.ratio_01 << "\n";
    out << "\tratio 10:    " << data_row.ratio_10 << "\n";
    
    return out;
}