#pragma once
#include <cstdint>
#include <string>
#include <fstream>
#include <limits>
#include <iostream>
#include <sstream>
#include <optional>
#include <exception>
#include "data_row.h"

class DataManager {
private:
    std::ifstream trades_file;
    std::ifstream tickers_file;

    DataRow data_row;
    
    struct TradeRow {
        std::string instr;
        int64_t timestamp = -1;
        double price;
        double vol;
        bool buyer_maker;
        
        void Update(DataRow& data_row) {
            if (instr == "btcusdt_fut") {
                if (buyer_maker) {
                    data_row.bid_pr_0 = price;
                } else {
                    data_row.ask_pr_0 = price;
                }
            } else if (instr == "btcbusd_fut") {
                if (buyer_maker) {
                    data_row.bid_pr_1 = price;
                } else {
                    data_row.ask_pr_1 = price;
                }
            } else {
                std::cerr << "\nTradeRow: unexpected instrument: " << instr << "\n";
            }
        }
    };

    struct TickerRow {
        std::string instr;
        int64_t timestamp = -1;
        double bid_pr;
        double bid_vol;
        double ask_pr;
        double ask_vol;

        bool usdt_ticker_valid = false;
        bool busd_ticker_valid = false;
        bool ratio_ticker_valid = false;

        void Update(DataRow& data_row) {
            if (instr == "btcusdt_fut") {
                data_row.bid_pr_0 = bid_pr;
                data_row.ask_pr_0 = ask_pr;
                data_row.bid_vol_0 = bid_vol;
                data_row.ask_vol_0 = ask_vol;

                usdt_ticker_valid = true;
            } else if (instr == "btcbusd_fut") {
                data_row.bid_pr_1 = bid_pr;
                data_row.ask_pr_1 = ask_pr;
                data_row.bid_vol_1 = bid_vol;
                data_row.ask_vol_1 = ask_vol;

                busd_ticker_valid = true;
            } else if (instr == "busdusdt") {
                data_row.ratio_01 = bid_pr;
                data_row.ratio_10 = 1 / ask_pr;

                ratio_ticker_valid = true;
            } else {
                std::cerr << "\nTickerRow: unexpected instrument: " << instr << "\n";
            }
        }
    };

    TradeRow trade_row;
    TickerRow ticker_row;

public:
    DataManager(const std::string& trades_path, const std::string& tickers_path) : 
    trades_file(trades_path), tickers_file(tickers_path) {
        if (trades_file.is_open()) {
            std::cout << "trades file opened successfully\n";
        } else {
            std::cout << "trades file is not open\n";
            std::cout << "\tpath: " << trades_path << "\n";
            throw std::invalid_argument("\ntrades file is not open\n");
        }

        if (tickers_file.is_open()) {
            std::cout << "tickers file opened successfully\n";
        } else {
            std::cout << "tickers file is not open\n";
            std::cout << "\tpath: " << tickers_path << "\n";
            throw "tickers file is not open\n";
        }

        data_row.timestamp = -1;
        std::string first_row;
        std::getline(tickers_file, first_row);
        if (first_row != ",instr,exch_ts,bid_pr,bid_vol,ask_pr,ask_vol") {
            std::cerr << 
                "incorrect data format:\n\texpected first line: \",instr,exch_ts,bid_pr,bid_vol,ask_pr,ask_vol\"\n";
        }

        std::getline(trades_file, first_row);
        if (first_row != ",instr,exch_ts,price,vol,buyer_maker") {
            std::cerr << 
                "incorrect data format:\n\texpected first line: \",instr,exch_ts,price,vol,buyer_maker\"\n";
        }
    }

    std::optional<DataRow> GetData(int64_t timestamp) {
        while(trade_row.timestamp <= timestamp || ticker_row.timestamp <= timestamp) {
            if (trade_row.timestamp < ticker_row.timestamp) {
                if (trade_row.timestamp != -1) {
                    trade_row.Update(data_row);
                }
                
                int64_t index;
                if (!(trades_file >> index)) {
                    data_row.timestamp = std::numeric_limits<int64_t>::infinity();
                    // std::cerr << "trades finished\n";
                    break;
                }

                char comma;
                trades_file >> comma;
                std::getline(trades_file, trade_row.instr, ',');
                trades_file >> trade_row.timestamp >> comma;
                trades_file >> trade_row.price >> comma;
                trades_file >> trade_row.vol >> comma;
                trades_file >> trade_row.buyer_maker;

            } else {
                if (ticker_row.timestamp != -1) {
                    ticker_row.Update(data_row);
                }

                int64_t index;
                if (!(tickers_file >> index)) {
                    data_row.timestamp = std::numeric_limits<double>::infinity();
                    // std::cerr << "tickers finished\n";
                    break;
                }

                char comma;
                tickers_file >> comma;
                std::getline(tickers_file, ticker_row.instr, ',');
                tickers_file >> ticker_row.timestamp >> comma;
                tickers_file >> ticker_row.bid_pr >> comma;
                tickers_file >> ticker_row.bid_vol >> comma;
                tickers_file >> ticker_row.ask_pr >> comma;
                tickers_file >> ticker_row.ask_vol;

            }
        }

        data_row.timestamp = timestamp;

        if (ticker_row.usdt_ticker_valid && ticker_row.busd_ticker_valid && ticker_row.ratio_ticker_valid) {
            return data_row;
        } else {
            return {};
        }
    }
};
