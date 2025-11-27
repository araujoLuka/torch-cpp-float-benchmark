#pragma once
#include <limits>

class EarlyStopping {
public:
    EarlyStopping(int patience)
        : patience_(patience),
          best_loss_(std::numeric_limits<double>::infinity()),
          counter_(0),
          stop_(false) {}

    const bool operator()(const double val_loss) {
        const double adjusted_best = best_loss_ - k_min_delta_;
        if (val_loss < adjusted_best) {
            best_loss_ = val_loss;
            counter_ = 0;
        } else {
            counter_++;
            if (counter_ >= patience_) {
                stop_ = true;
            }
        }
        return stop_;
    }

    const double best_loss() const { return best_loss_; }
    const bool should_stop() const { return stop_; }

private:
    static constexpr float k_min_delta_ = 2e-3;

    const int patience_;
    double best_loss_;
    int counter_;
    bool stop_;
};
