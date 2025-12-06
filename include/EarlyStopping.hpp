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
        const double adjusted_best = this->best_loss_ - k_min_delta_;
        if (val_loss < adjusted_best) {
            this->best_loss_ = val_loss;
            this->counter_ = 0;
        } else {
            this->counter_++;
            if (this->counter_ >= this->patience_) {
                this->stop_ = true;
            }
        }
        return this->stop_;
    }

    double best_loss() const { return best_loss_; }
    bool should_stop() const { return stop_; }
    
private:
    static constexpr float k_min_delta_ = 2e-3;

    int patience_;
    double best_loss_;
    int counter_;
    bool stop_;
};
