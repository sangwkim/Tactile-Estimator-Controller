#include <cpp/greeting.h>
#include <cpp/PenEven.h>
#include <cpp/TorqPoint.h>
#include <cpp/ContactMotion.h>
#include <cpp/PoseDiff.h>
#include <cpp/PenHinge.h>
#include <cpp/DispDiff.h>
#include <cpp/Wrench.h>
#include <cpp/WrenchInc.h>
#include <cpp/DispVar.h>
#include <cpp/EnergyElastic.h>
#include <cpp/TorqLine.h>

// The namespace should be the same as in the c++ source code.
namespace gtsam_custom_factors {

virtual class EnergyElastic : gtsam::NoiseModelFactor {
  EnergyElastic(size_t key1, size_t key2, double cost_sigma);
};

virtual class DispVar : gtsam::NoiseModelFactor {
  DispVar(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model);
};

virtual class WrenchInc : gtsam::NoiseModelFactor {
  WrenchInc(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class Wrench : gtsam::NoiseModelFactor {
  Wrench(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5,
    const gtsam::noiseModel::Base* model);
};

virtual class PenEven : gtsam::NoiseModelFactor {
  PenEven(size_t key1, size_t key2, size_t key3,
    const gtsam::noiseModel::Base* model);
};

virtual class TorqPoint : gtsam::NoiseModelFactor {
  TorqPoint(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Vector6& v_nominal,
    const gtsam::noiseModel::Base* model);
};

virtual class TorqLine : gtsam::NoiseModelFactor {
  TorqLine(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Vector6& v_nominal,
    const gtsam::noiseModel::Base* model);
};

virtual class ContactMotion : gtsam::NoiseModelFactor {
  ContactMotion(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class PoseDiff : gtsam::NoiseModelFactor {
  PoseDiff(size_t key1, size_t key2, size_t key3,
    const gtsam::noiseModel::Base* model, bool zeroJac);
};

virtual class PenHinge : gtsam::NoiseModelFactor {
  PenHinge(size_t key1, size_t key2, size_t key3,
    double cost_sigma, double eps);
};

virtual class DispDiff : gtsam::NoiseModelFactor {
  DispDiff(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose3& m, const gtsam::noiseModel::Base* model, bool zeroJac);
};

class Greeting {
  Greeting();
  void sayHello() const;
  gtsam::Rot3 invertRot3(gtsam::Rot3 rot) const;
  void sayGoodbye() const;
};

}  // namespace gtsam_example
