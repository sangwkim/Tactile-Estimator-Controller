#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class ExtrinsicStiff: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> {

public:

  ExtrinsicStiff(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3>(model, key1, key2, key3) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& p1,
            const gtsam::Pose3& p2,
            const gtsam::Vector3& s,
            boost::optional<gtsam::Matrix&> H1 = boost::none,
            boost::optional<gtsam::Matrix&> H2 = boost::none,
            boost::optional<gtsam::Matrix&> Hs = boost::none) const {
      
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H1_, H2_;

      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(p1,p2,&H1_,&H2_);
      gtsam::Vector e = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished();

      gtsam::Matrix Hes = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished().asDiagonal();

      gtsam::Vector output = (gtsam::Vector2() << lm[3]/e[2], lm[5]/e[2]).finished();

      gtsam::Matrix Hlm = (gtsam::Matrix26() << 0, 0, 0, 1/e[2], 0, 0,
                                                0, 0, 0, 0, 0, 1/e[2]).finished();
      gtsam::Matrix Hs_ = (gtsam::Matrix23() << 0, 0, -lm[3]/e[2]/e[2],
                                                0, 0, -lm[5]/e[2]/e[2]).finished();

      //if (H1) *H1 = Hlm * H1_;
      if (H1) *H1 = gtsam::Matrix::Zero(2,6);
      if (H2) *H2 = Hlm * H2_;
      if (Hs) *Hs = gtsam::Matrix::Zero(2,3); // Hs_ * Hes;

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class ExtrinsicStiff

} /// namespace gtsam_custom_factors