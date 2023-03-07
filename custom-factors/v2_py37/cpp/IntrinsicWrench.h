/*
((p1^(-1) * p3)^(-1) * (p2^(-1) * p4))
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class IntrinsicWrench: public gtsam::NoiseModelFactor6<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6> {

private:

    double N;

public:

  IntrinsicWrench(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, gtsam::Key key6, gtsam::SharedNoiseModel model, const double step_num) :
      gtsam::NoiseModelFactor6<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6>(model, key1, key2, key3, key4, key5, key6), N(step_num) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& p1,
            const gtsam::Pose3& p2,
            const gtsam::Pose3& p3,
            const gtsam::Pose3& p4,
            const gtsam::Vector3& s,
            const gtsam::Vector6& w,
            boost::optional<gtsam::Matrix&> H1 = boost::none,
            boost::optional<gtsam::Matrix&> H2 = boost::none,
            boost::optional<gtsam::Matrix&> H3 = boost::none,
            boost::optional<gtsam::Matrix&> H4 = boost::none,
            boost::optional<gtsam::Matrix&> Hs = boost::none,
            boost::optional<gtsam::Matrix&> Hw = boost::none) const {
      
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H1_, H2_, H3_, H4_, H13, H24;

      gtsam::Pose3 p13 = gtsam::traits<gtsam::Pose3>::Between(p1,p3,&H1_,&H3_);
      gtsam::Pose3 p24 = gtsam::traits<gtsam::Pose3>::Between(p2,p4,&H2_,&H4_);
      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(p13,p24,&H13,&H24);
      gtsam::Vector e = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished();
      gtsam::Vector wrench_gf = (gtsam::Vector6() << 0, 0, lm[2]/e[0]/e[0], lm[3]/e[1]/e[1], lm[4]/e[1]/e[1], 0).finished(); // gripper frame wrench
      
      gtsam::Vector torq_wf = p3.rotation().rotate( (gtsam::Vector3() << 0, 0, lm[2]/e[0]/e[0]).finished() );
      gtsam::Vector force_wf = p3.rotation().rotate( (gtsam::Vector3() << lm[3]/e[1]/e[1], lm[4]/e[1]/e[1], 0).finished() );      
      gtsam::Vector wrench_wf = (gtsam::Vector6() << torq_wf[0], torq_wf[1], torq_wf[2], force_wf[0], force_wf[1], force_wf[2]).finished();

      gtsam::Vector output = w - 0.01*wrench_wf/N;

      if (H1) *H1 = gtsam::Matrix::Zero(6,6);
      if (H2) *H2 = gtsam::Matrix::Zero(6,6);
      if (H3) *H3 = gtsam::Matrix::Zero(6,6);
      if (H4) *H4 = gtsam::Matrix::Zero(6,6);
      if (Hs) *Hs = gtsam::Matrix::Zero(6,3);
      if (Hw) *Hw = (gtsam::Vector6() << 1, 1, 1, 1, 1, 1).finished().asDiagonal();

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 6;
  }

}; // \class IntrinsicWrench

} /// namespace gtsam_custom_factors