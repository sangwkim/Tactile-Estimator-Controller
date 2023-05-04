/*
(measurement)^(-1) * ((p1^(-1) * p3)^(-1) * (p2^(-1) * p4))
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class TagCalibration: public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

private:

  gtsam::Pose3 p_wg_, p_go_, p_cam_tag_;

public:

  TagCalibration(gtsam::Key key1, gtsam::Key key2,
    const gtsam::Pose3 p_wg, const gtsam::Pose3 p_go, const gtsam::Pose3 p_cam_tag, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, key1, key2),
      p_wg_(p_wg), p_go_(p_go), p_cam_tag_(p_cam_tag) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p_w_cam, const gtsam::Pose3& p_tag_o,
    boost::optional<gtsam::Matrix&> H_w_cam = boost::none, boost::optional<gtsam::Matrix&> H_tag_o = boost::none) const {

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H_w_cam_, H_tag_o_, H_w_tag, H_wo, H_dummy;

      gtsam::Pose3 p_wo_ = gtsam::traits<gtsam::Pose3>::Compose(p_wg_, p_go_);
      gtsam::Pose3 p_w_tag = gtsam::traits<gtsam::Pose3>::Compose(p_w_cam, p_cam_tag_, &H_w_cam_, &H_dummy);
      gtsam::Pose3 p_wo = gtsam::traits<gtsam::Pose3>::Compose(p_w_tag, p_tag_o, &H_w_tag, &H_tag_o_);

      gtsam::Vector output = gtsam::traits<gtsam::Pose3>::Local(p_wo_, p_wo, &H_dummy, &H_wo);

      if (H_w_cam) *H_w_cam =  H_wo * H_w_tag * H_w_cam_;
      if (H_tag_o) *H_tag_o = H_wo * H_tag_o_;

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class TagCalibration

} /// namespace gtsam_custom_factors