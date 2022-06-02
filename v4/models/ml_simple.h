#pragma once
#include "IModel.h"
#include "../solvers/ema.h"
#include "../solvers/parameters_parser.h"
#include <fstream>

class ML_Simple : public IModel {
    double comission;
    double max_balance;

    EMA mean_convert0;
    EMA mean_convert1;

public:
    ML_Simple(const std::string& settings_path) {
        double mean_alpha;
        
        SET_VARIABLES(settings_path
            ,comission
            ,mean_alpha
            ,max_balance
        );

        mean_convert0.SetAlpha(mean_alpha);
        mean_convert1.SetAlpha(mean_alpha);
    }

    void Update(const DataRow& data_row) {

        double convert0 = data_row.bid_pr_1 / data_row.ask_pr_0;
        double convert1 = data_row.bid_pr_0 / data_row.ask_pr_1;
        
        double limit_convert0 = data_row.bid_pr_1 / data_row.bid_pr_0;
        double limit_convert1 = data_row.bid_pr_0 / data_row.bid_pr_1;

        mean_convert0.Update(convert0);
        mean_convert1.Update(convert1);

        if (limit_convert0 * mean_convert1.Get() * (1 - comission) > 1.0 && 
            exchange->Balance0() > -max_balance) {
            exchange->LimitMarketOrder_0(1.0);
        }
        if (limit_convert1 * mean_convert0.Get() * (1 - comission) > 1.0 && 
            exchange->Balance1() > -max_balance) {
            exchange->LimitMarketOrder_1(1.0);
        }
    }
};  