#include <cpp/NOCFactor.h>
#include <cpp/StiffnessRatioFactor6.h>
#include <cpp/ContactMotionFactor.h>
#include <cpp/PoseOdometryFactor.h>
#include <cpp/PenetrationFactor_2.h>
#include <cpp/TactileTransformFactor_3D.h>
#include <cpp/WrenchFactor.h>
#include <cpp/WrenchPredictFactor.h>
#include <cpp/DisplacementFactor.h>
#include <cpp/WrenchEnergyFactor.h>
#include <cpp/StiffnessRatioLine.h>

// The namespace should be the same as in the c++ source code.
namespace gtsam_custom_factors {

virtual class WrenchEnergyFactor : gtsam::NoiseModelFactor {
  WrenchEnergyFactor(size_t key1, size_t key2, double cost_sigma);
};

virtual class DisplacementFactor : gtsam::NoiseModelFactor {
  DisplacementFactor(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model);
};

virtual class WrenchPredictFactor : gtsam::NoiseModelFactor {
  WrenchPredictFactor(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class WrenchFactor : gtsam::NoiseModelFactor {
  WrenchFactor(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5,
    const gtsam::noiseModel::Base* model);
};

virtual class NOCFactor : gtsam::NoiseModelFactor {
  NOCFactor(size_t key1, size_t key2, size_t key3,
    const gtsam::noiseModel::Base* model);
};

virtual class StiffnessRatioFactor6 : gtsam::NoiseModelFactor {
  StiffnessRatioFactor6(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Vector6& v_nominal,
    const gtsam::noiseModel::Base* model);
};

virtual class StiffnessRatioLine : gtsam::NoiseModelFactor {
  StiffnessRatioLine(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Vector6& v_nominal,
    const gtsam::noiseModel::Base* model);
};

virtual class ContactMotionFactor : gtsam::NoiseModelFactor {
  ContactMotionFactor(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class PoseOdometryFactor : gtsam::NoiseModelFactor {
  PoseOdometryFactor(size_t key1, size_t key2, size_t key3,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class PenetrationFactor_2 : gtsam::NoiseModelFactor {
  PenetrationFactor_2(size_t key1, size_t key2, size_t key3,
    double cost_sigma, double eps);
};

virtual class TactileTransformFactor_3D : gtsam::NoiseModelFactor {
  TactileTransformFactor_3D(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose3& m, const gtsam::noiseModel::Base* model, bool zeroJac);
};

}
