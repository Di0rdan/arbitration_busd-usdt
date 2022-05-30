#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/preprocessor.hpp>


class ParametersParser {
private:
    std::unordered_map<std::string, std::string> storage;

public:
    ParametersParser(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "ParametersParser::ParametersParser:\n"
                         "\tcannot open file: " << file_path << "\n";
        }
        std::string line;
        while (std::getline(file, line)) {
            std::string key, value;
            // std::cout << line << "\n";
            std::stringstream sline(line);
            std::string variable_name;
            sline >> key >> value;
            storage[key] = value;

            // std::cout << "'" << key << "' -> '" << value << "'\n";
        }
        // std::cout << std::endl;
    }

    template<typename T>
    void Get(const std::string& key, T* value) {
        // std::cout << "Get(" << key << ", " << typeid(value).name() << "\n";
        if (storage.find(key) != storage.end()) {
            std::stringstream(storage[key]) >> *value;
        }
    }

    template<>
    void Get<int>(const std::string& key, int* value) {
        // std::cout << "Get(" << key << ", " << typeid(value).name() << "\n";
        if (storage.find(key) != storage.end()) {
            *value = std::stoi(storage[key]);
        }
    }

    template<>
    void Get<double>(const std::string& key, double* value) {
        // std::cout << "Get(" << key << ", " << typeid(value).name() << "\n";
        if (storage.find(key) != storage.end()) {
            *value = std::stod(storage[key]);
        }
    }
};

#define PROCESS_ONE_ELEMENT(r, unused, idx, elem) \
    BOOST_PP_COMMA_IF(idx) std::pair(BOOST_PP_STRINGIZE(elem), &elem)

#define SET_VARIABLES(path, ...) \
    setVariables(path, BOOST_PP_SEQ_FOR_EACH_I(PROCESS_ONE_ELEMENT, %%, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))


template<class... Args>
void setVariables(const std::string& path, Args... args) {
    std::cerr << path << ":\n";
    ParametersParser parser(path);
    (parser.Get(args.first, args.second), ...);
    (   
        (void)
        (
            std::cerr << "\t" << args.first << ":"
            << std::string(
                std::max(0, 20 - (int)(std::string(args.first).size())), 
                ' ')
            << (*(args.second)) << "\n"
        )
        , ...
    );
}
