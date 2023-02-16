/*
p1 (G(i-1)), p2 (G(i)), p3 (C(i-1)) --> Local Motion at the Estimated Point Contact
p4 --> Variable simply representing the above Local Motion
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class ContactMotion: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:
  bool zj;

public:

  ContactMotion(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::SharedNoiseModel model, bool zeroJac) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3, key4), zj(zeroJac) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2, const gtsam::Pose3& p3, const::gtsam::Pose3& p4,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none,
    boost::optional<gtsam::Matrix&> H3 = boost::none, boost::optional<gtsam::Matrix&> H4 = boost::none) const {

      gtsam::Pose3 p13 = gtsam::traits<gtsam::Pose3>::Between(p1, p3, H1, H3);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H13;
      gtsam::Pose3 p213 = gtsam::traits<gtsam::Pose3>::Compose(p2, p13, H2, &H13);
      if (H1) *H1 = H13 * (*H1);
      if (H3) *H3 = H13 * (*H3);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H213;
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H3_;
      gtsam::Pose3 p3213 = gtsam::traits<gtsam::Pose3>::Between(p3, p213, &H3_, &H213);
      if (H1) *H1 = H213 * (*H1);
      if (H2) *H2 = H213 * (*H2);
      if (H3) *H3 = H3_ + H213 * (*H3);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H3213;
      gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(p3213, p4, &H3213, H4);
      if (H1) *H1 = H3213 * (*H1);
      if (H2) *H2 = H3213 * (*H2);
      if (H3) *H3 = H3213 * (*H3);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HLM;
      gtsam::Vector6 lm = gtsam::traits<gtsam::Pose3>::Logmap(hx, &HLM);
      if (zj==true) {
        if (H1) *H1 = gtsam::Matrix66::Zero();
      } else {
        if (H1) *H1 = HLM * (*H1);
      }
      if (H2) *H2 = HLM * (*H2);
      if (H3) *H3 = gtsam::Matrix66::Zero();//HLM * (*H3);
      if (H4) *H4 = HLM * (*H4);

      return lm;

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class ContactMotion

} /// namespace gtsam_custom_factors