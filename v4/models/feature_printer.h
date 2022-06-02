#pragma once
#include "IModel.h"
#include "I_ML_Model.h"
#include "features/feature_manager.h"
#include "../solvers/parameters_parser.h"
#include <fstream>
#include <iomanip>

class FeaturePrinter : public I_ML_Model {
private:
    std::ofstream out;
    std::string separator = ",";

    void Learn(const std::vector<double>& learn_features, const std::vector<double>& learn_targets) override {
        // std::cerr << "printing row...\n";

        // std::cerr << out.is_open() << "\n";

        for (int i = 0; i != learn_features.size(); ++i) {
            if (i) {
                out << separator;
            }
            out << learn_features[i];
        }
        for (int i = 0; i != learn_targets.size(); ++i) {
            if (learn_features.size()) {
                out << separator;
            }
            out << learn_targets[i];
        }
        out << "\n";
    }

    void Predict(const DataRow& data_row, const std::vector<double>&) override {
        
    }

public:
    // FeaturePrinter(const FeaturePrinter& other) :   
    // feature_manager(other.feature_manager),
    // separator(other.separator)
    // {}
    
    FeaturePrinter(const std::string& settings_path) : I_ML_Model(settings_path) {
        std::string output_path;

        SET_VARIABLES(settings_path
            ,output_path
            ,separator
        );
        
        if (output_path.size()) {
            out.open(output_path);
            if (!out.is_open()) {
                std::cerr << "cannot open output file: " << output_path << "\n";
            } else {
                bool list_started = false;
                for (const auto& name : feature_manager.GetFeaturesNames()) {
                    if (list_started)
                        out << separator;
                    out << name;
                    list_started = true;
                }
                for (const auto& name : feature_manager.GetTargetsNames()) {
                    if (list_started)
                        out << separator;
                    out << name;
                    list_started = true;
                }
                out << "\n";
                out << std::fixed << std::setprecision(10);
            }
        } else {
            std::cerr << "expected \"output_path\" parameter in settings file\n";
        }
    }
};
