#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>


#include "price_feature.h"
#include "mean_price_feature.h"

#include "volume_feature.h"
#include "mean_volume_feature.h"

#include "convert_feature.h"
#include "mean_convert_feature.h"

#include "targets/limit_price.h"

class FeatureManager {
private:
    std::vector<std::shared_ptr<IFeature>> features;
    std::vector<std::shared_ptr<ITarget>> targets;
    int64_t target_delay = 0;

    std::vector<std::string> feature_names, target_names;

    std::deque<std::vector<double>> unsent_features, unsent_targets;
    // std::deque<int64_t> unsent_timestamp;
    std::deque<std::vector<double>> unresolved_features, unresolved_targets;
    // std::deque<int64_t> unresolved_timestamp;

    int64_t timestamp;
    int max_target_tick_delay = 0;

public:
    void SetFeatures (const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "cannot open feature settings file:\n\t" << path << "\n";
        }

        features.resize(0);

        std::string line;
        while (std::getline(file, line)) {
            std::string name;
            std::string value;
        
            std::stringstream in_line(line);
            in_line >> name >> value;
            if (name == "feature") {
                bool valid_feature_name = true;
                if (value == "BidPrice0") {
                    features.push_back(std::make_shared<BidPrice0>());
                } else if (value == "AskPrice0") {
                    features.push_back(std::make_shared<AskPrice0>());
                } else if (value == "BidPrice1") {
                    features.push_back(std::make_shared<BidPrice1>());
                } else if (value == "AskPrice1") {
                    features.push_back(std::make_shared<AskPrice1>());
                } else if (value == "BidVolume0") {
                    features.push_back(std::make_shared<BidVolume0>());
                } else if (value == "AskVolume0") {
                    features.push_back(std::make_shared<AskVolume0>());
                } else if (value == "BidVolume1") {
                    features.push_back(std::make_shared<BidVolume1>());
                } else if (value == "AskVolume1") {
                    features.push_back(std::make_shared<AskVolume1>());
                } else if (value == "Convert0") {
                    features.push_back(std::make_shared<Convert0>());
                } else if (value == "Convert1") {
                    features.push_back(std::make_shared<Convert1>());
                } else if (value == "MeanBidPrice0") {
                    features.push_back(std::make_shared<MeanBidPrice0>());
                } else if (value == "MeanAskPrice0") {
                    features.push_back(std::make_shared<MeanAskPrice0>());
                } else if (value == "MeanBidPrice1") {
                    features.push_back(std::make_shared<MeanBidPrice1>());
                } else if (value == "MeanAskPrice1") {
                    features.push_back(std::make_shared<MeanAskPrice1>());
                } else if (value == "MeanBidVolume0") {
                    features.push_back(std::make_shared<MeanBidVolume0>());
                } else if (value == "MeanAskVolume0") {
                    features.push_back(std::make_shared<MeanAskVolume0>());
                } else if (value == "MeanBidVolume1") {
                    features.push_back(std::make_shared<MeanBidVolume1>());
                } else if (value == "MeanAskVolume1") {
                    features.push_back(std::make_shared<MeanAskVolume1>());
                } else if (value == "MeanConvert0") {
                    features.push_back(std::make_shared<MeanConvert0>());
                } else if (value == "MeanConvert1") {
                    features.push_back(std::make_shared<MeanConvert1>());
                } else {
                    valid_feature_name = false;
                    std::cerr << "unknown feature name: " << value << "\n";
                }
                if (valid_feature_name) {
                    feature_names.push_back(value);
                }
                std::cerr << value << "\n";
            } else if (name.size()) {
                if (features.empty()) {
                    std::cerr << "parametr without feature:\n\t" << line << "\n";
                } else {
                    std::cerr << "\tset parametr: " << name << "=" << std::stod(value) << std::endl;
                    features[features.size() - 1]->SetParameter(name, std::stod(value));
                }
            }
        }
    }
    
    void SetTargets (const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "cannot open feature settings file:\n\t" <<
                path << "\n";
        }

        targets.resize(0);

        std::string line;
        while (std::getline(file, line)) {
            std::string name;
            std::string value;
        
            std::stringstream in_line(line);
            in_line >> name >> value;
            if (name == "target") {
                bool valid_target_name = true;
                if (value == "LimitPriceBid0") {
                    targets.push_back(std::make_shared<LimitPriceBid0>());
                } else if (value == "LimitPriceBid1") {
                    targets.push_back(std::make_shared<LimitPriceBid1>());
                } else {
                    valid_target_name = false;
                    std::cerr << "unknown feature name: " << value << "\n";
                }
                if (valid_target_name) {
                    target_names.push_back(value);
                }
                std::cerr << value << "\n";
            } else if (name.size()) {
                if (targets.empty()) {
                    std::cerr << "parametr without feature:\n\t" << line << "\n";
                } else {
                    std::cerr << "\tset parametr: " << name << "=" << std::stod(value) << std::endl;
                    targets[targets.size() - 1]->SetParameter(name, std::stod(value));
                }
            }
        }

        for (const auto& ptr : targets) {
            max_target_tick_delay = std::max(
                max_target_tick_delay, 
                ptr->tick_delay_count
            );
        }

        std::cerr << "max_target_tick_delay = " << max_target_tick_delay << "\n";
    }

    FeatureManager() {}
    FeatureManager(const std::string& features_path, const std::string& target_path) {
        SetFeatures(features_path);
        SetTargets(target_path);
    }

    const std::vector<std::string>& GetFeaturesNames() const {
        return feature_names;
    }
    const std::vector<std::string>& GetTargetsNames() const {
        return target_names;
    }
    
    void Update(const DataRow& data_row) {
        for (auto& ptr : features) {
            ptr->Update(data_row);
        }
        for (auto& ptr : targets) {
            ptr->Update(data_row);
        }
        timestamp = data_row.timestamp;
    }

    std::vector<double> GetFeatures() {
        // std::cerr << "FeatureManager::GetFeatures\n";
        std::vector<double> vec(features.size());
        for (int i = 0; i != vec.size(); ++i) {
            vec[i] = features[i]->Get();
        }

        unresolved_features.push_back(vec);
        unresolved_targets.push_back(GetTargets());
        // unresolved_timestamp.push_back(timestamp);

        return vec;
    }

    bool SetFeaturesTargets(
        std::vector<double>& features_vec, 
        std::vector<double>& targets_vec) {
        
        // std::cerr << "FeatureManager::SetFeaturesTargets\n";
        
        // std::cerr << "unresolved, unsent = (" << unresolved_features.size() << ", " << unsent_features.size() << ")\n";

        while (unresolved_features.size() > max_target_tick_delay) {
            // std::cerr << "FeatureManager::SetFeaturesTargets::while\n";
            unsent_features.push_back(unresolved_features[0]);
            unsent_targets.push_back(unresolved_targets[targets[0]->tick_delay_count]);

            unresolved_features.pop_front();
            unresolved_targets.pop_front();
        }

        if (unsent_features.empty()) {
            // std::cerr << "return false\n";
            return false;
        }

        // std::cerr << "set data, return true\n";

        features_vec = unsent_features.front();
        targets_vec = unsent_targets.front();

        unsent_features.pop_front();
        unsent_targets.pop_front();

        return true;
    }

    std::vector<double> GetTargets() {
        // std::cerr << "FeatureManager::GetTargets\n";
        std::vector<double> vec(targets.size());
        for (int i = 0; i != vec.size(); ++i) {
            vec[i] = targets[i]->Get();
        }
        return vec;
    }
};