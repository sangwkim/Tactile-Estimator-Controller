/*
((p1^(-1) * p3)^(-1) * (p2^(-1) * p4))
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class IntrinsicGrasp: public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> {

public:

  IntrinsicGrasp(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3>(model, key1, key2, key3, key4, key5) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& p1,
            const gtsam::Pose3& p2,
            const gtsam::Pose3& p3,
            const gtsam::Pose3& p4,
            const gtsam::Vector3& s,
            boost::optional<gtsam::Matrix&> H1 = boost::none,
            boost::optional<gtsam::Matrix&> H2 = boost::none,
            boost::optional<gtsam::Matrix&> H3 = boost::none,
            boost::optional<gtsam::Matrix&> H4 = boost::none,
            boost::optional<gtsam::Matrix&> Hs = boost::none) const {
      
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H1_, H2_, H3_, H4_, H13, H24;

      gtsam::Pose3 p13 = gtsam::traits<gtsam::Pose3>::Between(p1,p3,&H1_,&H3_);
      gtsam::Pose3 p24 = gtsam::traits<gtsam::Pose3>::Between(p2,p4,&H2_,&H4_);
      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(p13,p24,&H13,&H24);
      gtsam::Vector e = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished();

      gtsam::Matrix Hes = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished().asDiagonal();

      gtsam::Vector output = (gtsam::Vector3() << lm[2]/e[0], lm[3]/e[1], lm[4]/e[1]).finished();

      gtsam::Matrix Hlm = (gtsam::Matrix36() << 0, 0, 1/e[0], 0, 0, 0,
                                                0, 0, 0, 1/e[1], 0, 0,
                                                0, 0, 0, 0, 1/e[1], 0).finished();
      gtsam::Matrix Hs_ = (gtsam::Matrix33() << -lm[2]/e[0]/e[0], 0, 0,
                                                0, -lm[3]/e[1]/e[1], 0,
                                                0, -lm[4]/e[1]/e[1], 0).finished();

      if (H1) *H1 = Hlm * H13 * H1_;
      //if (H1) *H1 = gtsam::Matrix::Zero(3,6);
      if (H2) *H2 = Hlm * H24 * H2_;
      if (H3) *H3 = Hlm * H13 * H3_;
      //if (H3) *H3 = gtsam::Matrix::Zero(3,6);
      if (H4) *H4 = Hlm * H24 * H4_;
      if (Hs) *Hs = gtsam::Matrix::Zero(3,3); // Hs_ * Hes;

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class IntrinsicGrasp

} /// namespace gtsam_custom_factors