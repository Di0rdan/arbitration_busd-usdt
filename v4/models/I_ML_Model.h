#pragma once 
#include "IModel.h"
#include "features/feature_manager.h"
#include "../solvers/parameters_parser.h"
#include <fstream>
#include <iomanip>

class I_ML_Model : public IModel {
protected:
    FeatureManager feature_manager = FeatureManager();
    uint64_t step_count = 0;
    uint64_t history_step_count = 0;

    virtual void Learn (const std::vector<double>& learn_features, const std::vector<double>& learn_targets) = 0;
    virtual void Predict (const DataRow& data_row, const std::vector<double>&) = 0;

public:

    I_ML_Model(const std::string& settings_path) {
        std::string feature_settings_path;
        std::string target_settings_path;

        SET_VARIABLES(settings_path
            ,feature_settings_path
            ,target_settings_path
            ,history_step_count
        );

        if (feature_settings_path.size()) {
            feature_manager.SetFeatures(feature_settings_path);
        } else {
            std::cerr << "expected \"feature_settings_path\" parameter in settings file\n";
        }
        if (target_settings_path.size()) {
            feature_manager.SetTargets(target_settings_path);
        } else {
            std::cerr << "expected \"target_settings_path\" parameter in settings file\n";
        }
    }

    void Update(const DataRow& data_row) {
        // std::cerr << "FeaturePrinter::Update\n";
        feature_manager.Update(data_row);

        // std::vector<double> targets =
        //     feature_manager.GetTargets();

        if (++step_count > history_step_count) {
            Predict(data_row, feature_manager.GetFeatures());
        
            std::vector<double> learn_features;
            std::vector<double> learn_targets;
            while (feature_manager.SetFeaturesTargets(
                learn_features, learn_targets
            )) {
                Learn(learn_features, learn_targets);
            }
        }
    }  
};
