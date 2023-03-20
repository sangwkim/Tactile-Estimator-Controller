#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class MotionHinge: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

public:

  MotionHinge(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, double cost_sigma) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(1, cost_sigma), key1, key2, key3) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& pg,
            const gtsam::Pose3& pc,
            const gtsam::Pose3& pG,
            boost::optional<gtsam::Matrix&> Hg = boost::none,
            boost::optional<gtsam::Matrix&> Hc = boost::none,
            boost::optional<gtsam::Matrix&> HG = boost::none) const {

            typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HG_, HGgc;
            typename gtsam::Matrix36 HcGgc;
            gtsam::Matrix Hy = (gtsam::Matrix13() << 0,1,0).finished();

            gtsam::Pose3 pgc = gtsam::traits<gtsam::Pose3>::Between(pg,pc);
            gtsam::Pose3 pGgc = gtsam::traits<gtsam::Pose3>::Compose(pG,pgc,&HG_,boost::none);
            gtsam::Pose3 pcGgc = gtsam::traits<gtsam::Pose3>::Between(pc,pGgc,boost::none,&HGgc);
            gtsam::Vector dc = pcGgc.translation(&HcGgc);

            if (Hg) *Hg = gtsam::Matrix16::Zero();
            if (Hc) *Hc = gtsam::Matrix16::Zero();

            if (dc[1] <= 0) {
              if (HG) *HG = gtsam::Matrix16::Zero();
              return (gtsam::Vector1() << 0.0).finished();
            } else {
              if (HG) *HG = Hy * HcGgc * HGgc * HG_;
              return (gtsam::Vector1() << dc[1]).finished();
            }

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class MotionHinge

} /// namespace gtsam_custom_factors