#pragma once
#include "IModel.h"
#include "I_ML_Model.h"
#include "features/feature_manager.h"
#include "../solvers/parameters_parser.h"
#include "../solvers/ema.h"
#include <fstream>
#include <iomanip>
#include "../solvers/ELR.h"
#include <algorithm>

class LimitPricePredictor : public I_ML_Model {
private:

    ELR elr;

    double comission;
    double max_balance;

    EMA mean_convert0;
    EMA mean_convert1;

    void Learn(const std::vector<double>& learn_features, const std::vector<double>& learn_targets) override {
        // std::cerr << "LimitPricePredictor::Learn\n";

        elr.Update(learn_features, {learn_features[0] -  learn_targets[0], learn_features[1] -  learn_targets[1]});
    }

    void Predict(const DataRow& data_row, const std::vector<double>& features) override {
        // std::cerr << "LimitPricePredictor::Predict\n";

        std::vector<double> predict = elr.Predict(features);
        
        double shift0 = std::max(predict[0], 0.0);
        double shift1 = std::max(predict[1], 0.0);

        double predicted_bid_pr_0 = data_row.bid_pr_0 - shift0;
        double predicted_bid_pr_1 = data_row.bid_pr_1 - shift1;

        double convert0 = data_row.bid_pr_1 / data_row.ask_pr_0;
        double convert1 = data_row.bid_pr_0 / data_row.ask_pr_1;
        
        double limit_convert0 = data_row.bid_pr_1 / predicted_bid_pr_0;
        double limit_convert1 = data_row.bid_pr_0 / predicted_bid_pr_1;

        mean_convert0.Update(convert0);
        mean_convert1.Update(convert1);

        if (limit_convert0 * mean_convert1.Get() * (1 - comission) > 1.0 && 
            exchange->Balance0() > -max_balance) {
            std::cerr << "shift0 = " << shift0 << "\n";
            exchange->LimitMarketOrder_0(1.0, predicted_bid_pr_0);
        }
        if (limit_convert1 * mean_convert0.Get() * (1 - comission) > 1.0 && 
            exchange->Balance1() > -max_balance) {
            std::cerr << "shift1 = " << shift1 << "\n";
            exchange->LimitMarketOrder_1(1.0, predicted_bid_pr_1);
        }
    }

public:
    // FeaturePrinter(const FeaturePrinter& other) :   
    // feature_manager(other.feature_manager),
    // separator(other.separator)
    // {}
    
    LimitPricePredictor(const std::string& settings_path) : I_ML_Model(settings_path) {
        double elr_alpha = 1e-5;
        double mean_alpha = 1e-5;

        SET_VARIABLES(settings_path
            ,elr_alpha
            ,comission
            ,mean_alpha
            ,max_balance
        );

        mean_convert0.SetAlpha(mean_alpha);
        mean_convert1.SetAlpha(mean_alpha);

        elr.SetAlpha(elr_alpha);

        elr.SetSize(feature_manager.GetFeaturesNames().size(), feature_manager.GetTargetsNames().size());
    }
};
