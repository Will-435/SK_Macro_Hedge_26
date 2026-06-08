// rep_metrics.cpp
//
// Computes the quantitative figures used in Sections 4 and 5 of the SK Hynix
// hedge tear sheet, using ONLY the price series already produced by the
// investigation-1, investigation-2, investigation-3 and Master pipelines.
//
// Input  : rep_metrics_input.csv  (date + daily close levels, one column per
//          series: skhynix, kospi, krw, dxy, smh, gold, mag7). The CSV is a
//          plain extraction of the directory parquet files, no metrics in it.
// Output : printed report of correlations (Pearson and Spearman), the SK Hynix
//          beta to KOSPI, the minimum-variance short-KOSPI hedge, and the
//          unhedged vs hedged performance (annualised return, volatility,
//          Sharpe, maximum drawdown, structural hedge drag).
//
// All series are daily local-currency closes. Returns are daily log returns.
// Risk-free rate is taken as zero for the Sharpe ratio.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <iomanip>

static const double TRADING_DAYS_PER_YEAR = 252.0;

// Backtest window: every statistic is computed over the most-recent
// BACKTEST_WINDOW daily observations (the trailing 252 trading days / one year),
// not the full ~2,778-day history. Set this above the sample length to revert to
// the full sample. The rolling beta range uses a shorter window so there is
// intra-year variation to report inside the one-year backtest.
// The full 2,778-day sample will mix regimes and skew results.
static const int BACKTEST_WINDOW = 252;
static const int ROLLING_BETA_WINDOW = 63;

// Convert a level series into daily log returns.
std::vector<double> log_returns(const std::vector<double>& levels) {
    std::vector<double> returns;
    returns.reserve(levels.size());
    for (std::size_t index = 1; index < levels.size(); ++index) {
        returns.push_back(std::log(levels[index] / levels[index - 1]));
    }
    return returns;
}

double mean_of(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) total += value;
    return total / static_cast<double>(values.size());
}

double stdev_of(const std::vector<double>& values) {
    double average = mean_of(values);
    double sum_squared = 0.0;
    for (double value : values) sum_squared += (value - average) * (value - average);
    return std::sqrt(sum_squared / static_cast<double>(values.size() - 1));
}

double pearson(const std::vector<double>& first, const std::vector<double>& second) {
    double mean_first = mean_of(first);
    double mean_second = mean_of(second);
    double covariance = 0.0, variance_first = 0.0, variance_second = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        double deviation_first = first[index] - mean_first;
        double deviation_second = second[index] - mean_second;
        covariance += deviation_first * deviation_second;
        variance_first += deviation_first * deviation_first;
        variance_second += deviation_second * deviation_second;
    }
    return covariance / std::sqrt(variance_first * variance_second);
}

// Fractional ranks with ties averaged, so Spearman handles repeated values.
std::vector<double> ranks_of(const std::vector<double>& values) {
    std::size_t count = values.size();
    std::vector<std::size_t> order(count);
    for (std::size_t index = 0; index < count; ++index) order[index] = index;
    std::sort(order.begin(), order.end(),
              [&values](std::size_t left, std::size_t right) { return values[left] < values[right]; });
    std::vector<double> ranks(count);
    std::size_t position = 0;
    while (position < count) {
        std::size_t run_end = position;
        while (run_end + 1 < count && values[order[run_end + 1]] == values[order[position]]) ++run_end;
        double average_rank = 0.5 * (static_cast<double>(position) + static_cast<double>(run_end)) + 1.0;
        for (std::size_t fill = position; fill <= run_end; ++fill) ranks[order[fill]] = average_rank;
        position = run_end + 1;
    }
    return ranks;
}

double spearman(const std::vector<double>& first, const std::vector<double>& second) {
    return pearson(ranks_of(first), ranks_of(second));
}

double beta_of(const std::vector<double>& asset, const std::vector<double>& market) {
    double mean_asset = mean_of(asset);
    double mean_market = mean_of(market);
    double covariance = 0.0, variance_market = 0.0;
    for (std::size_t index = 0; index < asset.size(); ++index) {
        covariance += (asset[index] - mean_asset) * (market[index] - mean_market);
        variance_market += (market[index] - mean_market) * (market[index] - mean_market);
    }
    return covariance / variance_market;
}

// Geometric annualised return from a daily log-return stream.
double annualised_return(const std::vector<double>& returns) {
    double total_log = 0.0;
    for (double value : returns) total_log += value;
    double exponent = TRADING_DAYS_PER_YEAR / static_cast<double>(returns.size());
    return std::exp(total_log * exponent) - 1.0;
}

double annualised_vol(const std::vector<double>& returns) {
    return stdev_of(returns) * std::sqrt(TRADING_DAYS_PER_YEAR);
}

// Sharpe with a zero risk-free rate, arithmetic mean annualised.
double sharpe_ratio(const std::vector<double>& returns) {
    return (mean_of(returns) * TRADING_DAYS_PER_YEAR) / annualised_vol(returns);
}

// Maximum drawdown of the compounded wealth path implied by the returns.
double max_drawdown(const std::vector<double>& returns) {
    double wealth = 1.0, peak = 1.0, worst = 0.0;
    for (double value : returns) {
        wealth *= std::exp(value);
        if (wealth > peak) peak = wealth;
        double drawdown = wealth / peak - 1.0;
        if (drawdown < worst) worst = drawdown;
    }
    return worst;
}

// Annualised Sharpe of the hedged book (asset minus hedge_ratio*market), rf = 0.
double hedged_sharpe(const std::vector<double>& asset, const std::vector<double>& market, double hedge_ratio) {
    std::vector<double> book(asset.size());
    for (std::size_t index = 0; index < asset.size(); ++index)
        book[index] = asset[index] - hedge_ratio * market[index];
    return (mean_of(book) * TRADING_DAYS_PER_YEAR) / (stdev_of(book) * std::sqrt(TRADING_DAYS_PER_YEAR));
}

// Annualised variance of the hedged book.
double hedged_variance(const std::vector<double>& asset, const std::vector<double>& market, double hedge_ratio) {
    std::vector<double> book(asset.size());
    for (std::size_t index = 0; index < asset.size(); ++index)
        book[index] = asset[index] - hedge_ratio * market[index];
    double daily = stdev_of(book);
    return daily * daily * TRADING_DAYS_PER_YEAR;
}

// Gradient ASCENT on the hedged Sharpe over the hedge ratio, using a central
// difference gradient. Returns the Sharpe-maximising hedge ratio.
double ascend_sharpe(const std::vector<double>& asset, const std::vector<double>& market,
                     double start, double learning_rate, int iterations) {
    double hedge_ratio = start;
    const double step = 1e-5;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        double gradient = (hedged_sharpe(asset, market, hedge_ratio + step)
                           - hedged_sharpe(asset, market, hedge_ratio - step)) / (2.0 * step);
        hedge_ratio += learning_rate * gradient;
    }
    return hedge_ratio;
}

// Gradient DESCENT on the hedged variance over the hedge ratio. Returns the
// variance-minimising hedge ratio, which must equal the beta as a sanity check.
double descend_variance(const std::vector<double>& asset, const std::vector<double>& market,
                        double start, double learning_rate, int iterations) {
    double hedge_ratio = start;
    const double step = 1e-5;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        double gradient = (hedged_variance(asset, market, hedge_ratio + step)
                           - hedged_variance(asset, market, hedge_ratio - step)) / (2.0 * step);
        hedge_ratio -= learning_rate * gradient;
    }
    return hedge_ratio;
}

// Overlapping forward horizon-day log returns, stamped at the entry day, used
// for the four-week scenario analysis.
std::vector<double> forward_returns(const std::vector<double>& levels, int horizon) {
    std::vector<double> forward;
    for (std::size_t entry = 0; entry + static_cast<std::size_t>(horizon) < levels.size(); ++entry)
        forward.push_back(std::log(levels[entry + horizon] / levels[entry]));
    return forward;
}

// Linear-interpolated percentile of a sample.
double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    double position = quantile * static_cast<double>(values.size() - 1);
    std::size_t lower = static_cast<std::size_t>(position);
    double fraction = position - static_cast<double>(lower);
    if (lower + 1 < values.size()) return values[lower] * (1.0 - fraction) + values[lower + 1] * fraction;
    return values[lower];
}

// Minimum, mean and maximum of a rolling-window beta of asset on market.
void rolling_beta_range(const std::vector<double>& asset, const std::vector<double>& market,
                        int window, double& out_min, double& out_mean, double& out_max) {
    out_min = 1e9; out_max = -1e9; double sum = 0.0; int windows = 0;
    for (std::size_t end = window; end <= asset.size(); ++end) {
        double mean_asset = 0.0, mean_market = 0.0;
        for (std::size_t index = end - window; index < end; ++index) { mean_asset += asset[index]; mean_market += market[index]; }
        mean_asset /= window; mean_market /= window;
        double covariance = 0.0, variance = 0.0;
        for (std::size_t index = end - window; index < end; ++index) {
            covariance += (asset[index] - mean_asset) * (market[index] - mean_market);
            variance += (market[index] - mean_market) * (market[index] - mean_market);
        }
        double beta = covariance / variance;
        if (beta < out_min) out_min = beta;
        if (beta > out_max) out_max = beta;
        sum += beta; ++windows;
    }
    out_mean = sum / windows;
}

// Sign of the first-half minus second-half Pearson, used as the trend arrow.
std::string trend_arrow(const std::vector<double>& first, const std::vector<double>& second) {
    std::size_t midpoint = first.size() / 2;
    std::vector<double> first_early(first.begin(), first.begin() + midpoint);
    std::vector<double> second_early(second.begin(), second.begin() + midpoint);
    std::vector<double> first_late(first.begin() + midpoint, first.end());
    std::vector<double> second_late(second.begin() + midpoint, second.end());
    double early = pearson(first_early, second_early);
    double late = pearson(first_late, second_late);
    double change = std::fabs(late) - std::fabs(early);
    if (change > 0.05) return "tightening";
    if (change < -0.05) return "loosening";
    return "stable";
}

void report_pair(const std::string& label,
                 const std::vector<double>& first, const std::vector<double>& second) {
    std::cout << std::left << std::setw(34) << label
              << "Pearson " << std::showpos << std::fixed << std::setprecision(3) << pearson(first, second)
              << "   Spearman " << spearman(first, second)
              << std::noshowpos << "   trend " << trend_arrow(first, second) << "\n";
}

void report_performance(const std::string& label, const std::vector<double>& returns) {
    std::cout << std::left << std::setw(12) << label
              << " ann.return " << std::showpos << std::fixed << std::setprecision(3) << annualised_return(returns) * 100.0 << "%"
              << "   ann.vol " << std::noshowpos << annualised_vol(returns) * 100.0 << "%"
              << "   Sharpe " << std::showpos << sharpe_ratio(returns)
              << "   maxDD " << std::noshowpos << max_drawdown(returns) * 100.0 << "%\n";
}

int main() {
    std::ifstream input_file("rep_metrics_input.csv");
    if (!input_file.is_open()) {
        std::cerr << "Could not open rep_metrics_input.csv\n";
        return 1;
    }

    std::string header_line;
    std::getline(input_file, header_line);
    std::vector<std::string> column_names;
    {
        std::stringstream header_stream(header_line);
        std::string field;
        while (std::getline(header_stream, field, ',')) column_names.push_back(field);
    }

    std::map<std::string, std::vector<double>> levels;
    std::string data_line;
    while (std::getline(input_file, data_line)) {
        if (data_line.empty()) continue;
        std::stringstream data_stream(data_line);
        std::string field;
        std::size_t column_index = 0;
        while (std::getline(data_stream, field, ',')) {
            if (column_index > 0) levels[column_names[column_index]].push_back(std::stod(field));
            ++column_index;
        }
    }

    // Restrict every series to the trailing BACKTEST_WINDOW observations so the
    // whole report is a 252-day (one-year) backtest. One extra level is kept so
    // the first daily log return falls inside the window.
    for (auto& column : levels) {
        std::vector<double>& series = column.second;
        std::size_t keep = static_cast<std::size_t>(BACKTEST_WINDOW) + 1;
        if (series.size() > keep) series.erase(series.begin(), series.end() - keep);
    }

    std::vector<double> skhynix = log_returns(levels["skhynix"]);
    std::vector<double> kospi   = log_returns(levels["kospi"]);
    std::vector<double> krw     = log_returns(levels["krw"]);
    std::vector<double> dxy     = log_returns(levels["dxy"]);
    std::vector<double> gold    = log_returns(levels["gold"]);
    std::vector<double> mag7    = log_returns(levels["mag7"]);
    std::vector<double> smh     = log_returns(levels["smh"]);

    std::cout << "Sample: " << skhynix.size() << " daily observations\n\n";

    std::cout << "SECTION 5A  Key factor correlations (daily log returns)\n";
    report_pair("KRW x Semiconductor equity", krw, skhynix);
    report_pair("Semi cycle x AI capex (SKH x Mag7)", skhynix, mag7);
    report_pair("USD (DXY) x KRW", dxy, krw);
    report_pair("Gold x Portfolio (SK Hynix)", gold, skhynix);
    report_pair("Hedge (KOSPI) x Portfolio", kospi, skhynix);

    double hedge_beta = beta_of(skhynix, kospi);
    std::vector<double> hedged;
    hedged.reserve(skhynix.size());
    for (std::size_t index = 0; index < skhynix.size(); ++index)
        hedged.push_back(skhynix[index] - hedge_beta * kospi[index]);

    double variance_removed = pearson(skhynix, kospi) * pearson(skhynix, kospi);
    double hedge_drag = hedge_beta * annualised_return(kospi);

    std::cout << "\nSECTION 5B  Hedge sizing and performance\n";
    std::cout << "SK Hynix beta to KOSPI (min-variance short-hedge ratio): "
              << std::fixed << std::setprecision(2) << hedge_beta << "\n";
    std::cout << "Variance removed by the beta hedge (R^2): "
              << std::setprecision(1) << variance_removed * 100.0 << "%\n";
    std::cout << "Structural hedge drag (beta x KOSPI ann.return), the carry of the short: "
              << std::showpos << std::setprecision(2) << hedge_drag * 100.0 << "%\n" << std::noshowpos;
    report_performance("Unhedged", skhynix);
    report_performance("Hedged", hedged);

    // ---- Section 5C: gradient optimisation of the hedge ratio ----
    // Two gradient searches over the hedge ratio h of the book r_skh - h*r_kospi:
    // descent on variance (recovers the beta) and ascent on Sharpe. Then test
    // whether any h both cuts variance versus unhedged and lifts Sharpe.
    double minvar_ratio = descend_variance(skhynix, kospi, 0.0, 1.0, 20000);
    double maxsharpe_ratio = ascend_sharpe(skhynix, kospi, 0.0, 0.05, 100000);
    double mean_skh_daily = mean_of(skhynix), mean_kospi_daily = mean_of(kospi);
    double var_skh_daily = stdev_of(skhynix); var_skh_daily *= var_skh_daily;
    double var_kospi_daily = stdev_of(kospi); var_kospi_daily *= var_kospi_daily;
    double cov_daily = hedge_beta * var_kospi_daily;
    double analytic_maxsharpe = (mean_kospi_daily * var_skh_daily - mean_skh_daily * cov_daily)
                              / (mean_kospi_daily * cov_daily - mean_skh_daily * var_kospi_daily);

    std::cout << "\nSECTION 5C  Hedge-ratio optimisation (gradient methods)\n";
    std::cout << "Min-variance hedge ratio (gradient descent on variance): "
              << std::fixed << std::setprecision(3) << minvar_ratio
              << "   (analytic beta " << hedge_beta << ")\n";
    std::cout << "Max-Sharpe hedge ratio  (gradient ascent on Sharpe):     "
              << std::showpos << maxsharpe_ratio << std::noshowpos
              << "   (analytic " << std::showpos << analytic_maxsharpe << std::noshowpos << ")\n";
    std::cout << "Variance is below the unhedged book only for hedge ratios in (0, "
              << std::setprecision(2) << 2.0 * hedge_beta << ")\n";
    std::cout << "\n  hedge_ratio      Sharpe   ann_variance\n";
    for (double ratio : {0.0, maxsharpe_ratio, hedge_beta}) {
        std::cout << "  " << std::showpos << std::setw(9) << std::setprecision(3) << ratio << std::noshowpos
                  << std::setw(12) << std::setprecision(3) << hedged_sharpe(skhynix, kospi, ratio)
                  << std::setw(13) << std::setprecision(4) << hedged_variance(skhynix, kospi, ratio) << "\n";
    }
    bool win_win = (maxsharpe_ratio > 0.0 && maxsharpe_ratio < 2.0 * hedge_beta
                    && hedged_sharpe(skhynix, kospi, maxsharpe_ratio) > hedged_sharpe(skhynix, kospi, 0.0));
    std::cout << "\nIs there a hedge ratio that BOTH cuts variance and raises Sharpe? "
              << (win_win ? "YES" : "NO") << "\n";
    if (!win_win) {
        std::cout << "  Sharpe peaks at a NEGATIVE hedge ratio (adding KOSPI exposure), which raises variance.\n";
        std::cout << "  Every variance-reducing short hedge (h > 0) lowers Sharpe: the objectives conflict.\n";
    }

    // ---- Section 6: scenario analysis on four-week (20-day) forward returns ----
    // Scenarios are defined by the SK Hynix four-week return distribution: the
    // bottom 15% is Risk-Off, the top 15% Risk-On, the middle 70% Base. For
    // each bucket we report the conditional mean four-week move of the book
    // (SK Hynix), the short-KOSPI hedge payoff (-beta x KOSPI move), the net
    // hedged book, and how the other factors behave in the same windows.
    const int HORIZON = 20;
    std::vector<double> skh_forward  = forward_returns(levels["skhynix"], HORIZON);
    std::vector<double> kospi_forward = forward_returns(levels["kospi"],  HORIZON);
    std::vector<double> krw_forward  = forward_returns(levels["krw"],     HORIZON);
    std::vector<double> dxy_forward  = forward_returns(levels["dxy"],     HORIZON);
    std::vector<double> gold_forward = forward_returns(levels["gold"],    HORIZON);
    std::vector<double> mag7_forward = forward_returns(levels["mag7"],    HORIZON);

    double low_threshold = percentile(skh_forward, 0.15);
    double high_threshold = percentile(skh_forward, 0.85);

    struct Scenario { const char* name; double lower; double upper; };
    Scenario scenarios[3] = {
        {"Risk-Off", -1e9, low_threshold},
        {"Base", low_threshold, high_threshold},
        {"Risk-On", high_threshold, 1e9},
    };

    std::cout << "\nSECTION 6  Scenario analysis (four-week forward returns, hedge beta "
              << std::fixed << std::setprecision(2) << hedge_beta << ")\n";
    for (const Scenario& scenario : scenarios) {
        int count = 0;
        double sum_skh = 0, sum_kospi = 0, sum_krw = 0, sum_dxy = 0, sum_gold = 0, sum_mag7 = 0;
        for (std::size_t index = 0; index < skh_forward.size(); ++index) {
            if (skh_forward[index] > scenario.lower && skh_forward[index] <= scenario.upper) {
                ++count;
                sum_skh += skh_forward[index]; sum_kospi += kospi_forward[index];
                sum_krw += krw_forward[index]; sum_dxy += dxy_forward[index];
                sum_gold += gold_forward[index]; sum_mag7 += mag7_forward[index];
            }
        }
        double probability = 100.0 * count / static_cast<double>(skh_forward.size());
        double mean_skh = sum_skh / count, mean_kospi = sum_kospi / count;
        double mean_krw = sum_krw / count, mean_dxy = sum_dxy / count;
        double mean_gold = sum_gold / count, mean_mag7 = sum_mag7 / count;
        double hedge_payoff = -hedge_beta * mean_kospi;
        double net = mean_skh + hedge_payoff;
        std::cout << std::left << std::setw(9) << scenario.name
                  << " prob " << std::setprecision(0) << probability << "%"
                  << std::showpos << std::setprecision(1)
                  << "  | SKH " << mean_skh * 100 << "%  hedge " << hedge_payoff * 100
                  << "%  net " << net * 100 << "%"
                  << "  || KRW " << mean_krw * 100 << "%  DXY " << mean_dxy * 100
                  << "%  Gold " << mean_gold * 100 << "%  Mag7 " << mean_mag7 * 100 << "%\n"
                  << std::noshowpos;
    }

    double beta_min, beta_mean, beta_max;
    rolling_beta_range(skhynix, kospi, ROLLING_BETA_WINDOW, beta_min, beta_mean, beta_max);
    std::cout << "\nRolling " << ROLLING_BETA_WINDOW << "d SK Hynix/KOSPI beta within the window (sizing-rule range): min "
              << std::setprecision(2) << beta_min << "  mean " << beta_mean
              << "  max " << beta_max << "\n";

    std::vector<double> net_forward;
    for (std::size_t index = 0; index < skh_forward.size(); ++index)
        net_forward.push_back(skh_forward[index] - hedge_beta * kospi_forward[index]);
    std::cout << "Worst observed four-week move: SK Hynix "
              << std::showpos << std::setprecision(1) << percentile(skh_forward, 0.0) * 100 << "%"
              << "   hedged " << percentile(net_forward, 0.0) * 100 << "%\n" << std::noshowpos;

    // ---- Section 3C: current risk-regime classification (as of last observation) ----
    const int RV_WINDOW = 21;
    std::vector<double> rv_history;
    for (std::size_t end = RV_WINDOW; end <= skhynix.size(); ++end) {
        std::vector<double> window(skhynix.begin() + end - RV_WINDOW, skhynix.begin() + end);
        rv_history.push_back(stdev_of(window) * std::sqrt(TRADING_DAYS_PER_YEAR));
    }
    double current_rv = rv_history.back();
    int rv_below = 0;
    for (double value : rv_history) if (value <= current_rv) ++rv_below;
    double rv_percentile = 100.0 * rv_below / static_cast<double>(rv_history.size());

    const std::vector<double>& skh_levels = levels["skhynix"];
    double current_20d = std::log(skh_levels.back() / skh_levels[skh_levels.size() - 1 - HORIZON]);
    const std::vector<double>& mag7_levels = levels["mag7"];
    double current_mag7_20d = std::log(mag7_levels.back() / mag7_levels[mag7_levels.size() - 1 - HORIZON]);

    double current_beta = beta_of(skhynix, kospi);
    double current_corr = pearson(skhynix, kospi);

    const std::vector<double>& volume = levels["skh_volume"];
    int long_volume_days = std::min<int>(252, static_cast<int>(volume.size()));
    double sum_20 = 0, sum_long = 0;
    for (int i = 0; i < 20; ++i) sum_20 += volume[volume.size() - 1 - i];
    for (int i = 0; i < long_volume_days; ++i) sum_long += volume[volume.size() - 1 - i];
    double liquidity_ratio = (sum_20 / 20.0) / (sum_long / long_volume_days);

    std::cout << "\nSECTION 3C  Risk-regime classification (trailing 252-day window)\n";
    std::cout << "Vol regime    : 21d realised vol " << std::fixed << std::setprecision(1) << current_rv * 100
              << "% annualised, " << std::setprecision(0) << rv_percentile << "th percentile within the window\n";
    std::cout << "Risk appetite : trailing 4-week SK Hynix " << std::showpos << std::setprecision(1) << current_20d * 100
              << "% (risk-off <= " << low_threshold * 100 << "%, risk-on >= " << high_threshold * 100 << "%); Mag7 "
              << current_mag7_20d * 100 << "%\n" << std::noshowpos;
    std::cout << "Correlation   : 252-day KOSPI corr " << std::showpos << std::setprecision(2) << current_corr
              << ", beta " << current_beta << std::noshowpos << " (63d rolling beta range " << beta_min << "-" << beta_max << ")\n";
    std::cout << "Liquidity     : 20d avg volume / 252d avg volume = " << std::setprecision(2) << liquidity_ratio
              << " (" << std::showpos << std::setprecision(0) << (liquidity_ratio - 1) * 100 << "% vs 1yr average)\n"
              << std::noshowpos;

    // ---- Correlation matrix supporting Section 4A (Pearson, daily log returns) ----
    // SK Hynix is the target risk. The candidate-hedge underlyings are SK Hynix
    // itself (single-name puts), KOSPI (index puts / collar) and gold; USD/KRW,
    // DXY, US semis and Mag 7 are carried for the cross-hedge context in 4B.
    std::vector<std::pair<std::string, const std::vector<double>*>> matrix_assets = {
        {"SK Hynix", &skhynix}, {"KOSPI", &kospi}, {"Gold", &gold},
        {"USD/KRW", &krw}, {"DXY", &dxy}, {"US semis", &smh}, {"Mag 7", &mag7},
    };
    std::ofstream matrix_file("rep_corr_matrix.csv");
    matrix_file << "asset";
    for (const auto& asset : matrix_assets) matrix_file << "," << asset.first;
    matrix_file << "\n";
    std::cout << "\nCORRELATION MATRIX (Pearson, daily log returns) — supports Section 4A\n";
    std::cout << std::setw(10) << " ";
    for (const auto& asset : matrix_assets) std::cout << std::setw(9) << asset.first;
    std::cout << "\n";
    for (const auto& row_asset : matrix_assets) {
        matrix_file << row_asset.first;
        std::cout << std::left << std::setw(10) << row_asset.first;
        for (const auto& col_asset : matrix_assets) {
            double correlation = pearson(*row_asset.second, *col_asset.second);
            matrix_file << "," << std::fixed << std::setprecision(3) << correlation;
            std::cout << std::showpos << std::fixed << std::setprecision(2) << std::setw(9) << correlation;
        }
        matrix_file << "\n";
        std::cout << std::noshowpos << "\n";
    }
    matrix_file.close();

    return 0;
}
