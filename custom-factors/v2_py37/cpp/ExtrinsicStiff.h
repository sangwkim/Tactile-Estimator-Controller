#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class ExtrinsicStiff: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6> {

private:

  bool zj;

public:

  ExtrinsicStiff(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::SharedNoiseModel model, bool zeroJac) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6>(model, key1, key2, key3, key4), zj(zeroJac) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& p1,
            const gtsam::Pose3& p2,
            const gtsam::Vector3& s,
            const gtsam::Vector6& w,
            boost::optional<gtsam::Matrix&> H1 = boost::none,
            boost::optional<gtsam::Matrix&> H2 = boost::none,
            boost::optional<gtsam::Matrix&> Hs = boost::none,
            boost::optional<gtsam::Matrix&> Hw = boost::none) const {
      
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H1_, H2_;

      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(p1,p2,&H1_,&H2_);
      gtsam::Vector e = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished();

      gtsam::Vector output = (gtsam::Vector2() << pow(std::abs(w[4]),0.5)*lm[3]/e[2], pow(std::abs(w[4]),0.5)*lm[5]/e[2]).finished();

      gtsam::Matrix Hlm = (gtsam::Matrix26() << 0, 0, 0, pow(std::abs(w[4]),0.5)/e[2], 0, 0,
                                                0, 0, 0, 0, 0, pow(std::abs(w[4]),0.5)/e[2]).finished();

      if (zj==true) {
        if (H1) *H1 = gtsam::Matrix::Zero(2,6);
      } else {
        if (H1) *H1 = Hlm * H1_;
      }      
      if (H2) *H2 = Hlm * H2_;
      if (Hs) *Hs = gtsam::Matrix::Zero(2,3);
      if (Hw) *Hw = gtsam::Matrix::Zero(2,6);

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class ExtrinsicStiff

} /// namespace gtsam_custom_factors