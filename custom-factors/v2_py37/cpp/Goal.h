/*
Encourage p1^(-1) * p2 == p3
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class Goal: public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

private:

  gtsam::Pose3 goal_;
  bool zj;

public:

  Goal(gtsam::Key key1, gtsam::Key key2, const gtsam::Pose3 goal, gtsam::SharedNoiseModel model, bool zeroJac) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, key1, key2), goal_(goal), zj(zeroJac) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& po, const gtsam::Pose3& pg, 
    boost::optional<gtsam::Matrix&> Ho = boost::none, boost::optional<gtsam::Matrix&> Hg = boost::none) const {

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hog;
      gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po, pg, Ho, Hg);
      gtsam::Vector output = gtsam::traits<gtsam::Pose3>::Local(goal_, pog, boost::none, &Hog);

      if (Ho) *Ho = gtsam::Matrix66::Zero();
      if (Hg) *Hg = Hog * (*Hg);
      
      return output;

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class Goal

} /// namespace gtsam_custom_factors