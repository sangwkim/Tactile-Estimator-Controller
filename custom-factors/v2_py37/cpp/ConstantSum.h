#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class ConstantSum: public gtsam::NoiseModelFactor1<gtsam::Vector3> {

private:

  double s_;

public:

  ConstantSum(gtsam::Key key1, const double sum, gtsam::SharedNoiseModel model) :
      s_(sum), gtsam::NoiseModelFactor1<gtsam::Vector3>(model, key1) {}

  gtsam::Vector evaluateError(const gtsam::Vector3& v,
    boost::optional<gtsam::Matrix&> Hv = boost::none) const {

      if (Hv) *Hv = (gtsam::Matrix(1,3) << 1, 1, 1).finished();
      return (gtsam::Vector(1) << v[0] + v[1] + v[2] - s_).finished();
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 1;
  }

}; // \class ConstantSum

} /// namespace gtsam_custom_factors