#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class PoseOdometryFactor: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:
  bool zj;

public:

  PoseOdometryFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::SharedNoiseModel model, bool zeroJac) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3), zj(zeroJac) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2, const gtsam::Pose3& p3, 
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none,
    boost::optional<gtsam::Matrix&> H3 = boost::none) const {

      gtsam::Pose3 p12 = gtsam::traits<gtsam::Pose3>::Between(p1, p2, H1, H2);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H12;
      gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(p12, p3, &H12, H3);
      if (H1) *H1 = H12 * (*H1);
      if (H2) *H2 = H12 * (*H2);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HLM;
      gtsam::Vector6 lm = gtsam::traits<gtsam::Pose3>::Logmap(hx, &HLM);
      if (zj==true) {
        if (H1) *H1 = gtsam::Matrix66::Zero();
      } else {
        if (H1) *H1 = HLM * (*H1);
      }
      if (H2) *H2 = HLM * (*H2);
      if (H3) *H3 = HLM * (*H3);

      return lm;

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class PoseOdometryFactor

} /// namespace gtsam_packing