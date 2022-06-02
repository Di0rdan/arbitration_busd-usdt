#include <Eigen/Dense>
#include <vector>

class ELR {
public:
    Eigen::MatrixXd XTX;
    Eigen::MatrixXd XTy;

    size_t n = 0, m = 0;

    double alpha = 1e-5;

public:
    void SetSize(size_t new_n, size_t new_m) {
        n = new_n;
        m = new_m;

        XTX.resize(n, n);
        XTy.resize(n, m);

        XTX *= 0;
        for (int i = 0; i != n; ++i) {
            XTX(i, i) = 1.0;
        }
        XTy *= 0;
    }

    void SetAlpha(double value) {
        alpha = value;
    }

    void Update(
        const std::vector<double>& features, 
        const std::vector<double>& targets) {
        
        Eigen::MatrixXd x(1, n);
        Eigen::MatrixXd y(1, m);

        for (size_t i = 0; i != features.size(); ++i) {
            x(0, i) = features[i];
        }

        for (size_t i = 0; i != targets.size(); ++i) {
            y(0, i) = targets[i];
        }

        x *= sqrt(alpha);
        y *= sqrt(alpha);

        XTX /= (1 - alpha);
        Eigen::MatrixXd XT_XTX = x * XTX;
        double scalar = sqrt(1.0 + (XT_XTX * x.transpose())(0, 0));
        XT_XTX /= scalar;
        XTX -= XT_XTX.transpose() * XT_XTX;

        XTy *= (1 - alpha);
        XTy += x.transpose() * y;
    }

    std::vector<double> Predict(const std::vector<double>& features ) {
        Eigen::MatrixXd x(1, n);
        for (size_t i = 0; i != n; ++i) {
            x(0, i) = features[i];
        }

        Eigen::MatrixXd y_pred = (x * XTX) * XTy;
        std::vector<double> predict(m);
        for (int i = 0; i != m; ++i) {
            predict[i] = y_pred(0, i);
        }

        return predict;
    }
};