#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class MotionHinge: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

    gtsam::Point3 oc;

public:

  MotionHinge(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, const gtsam::Point3 contact_point, double cost_sigma) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(1, cost_sigma), key1, key2, key3, key4),
      oc(contact_point) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& pg,
            const gtsam::Pose3& po,
            const gtsam::Pose3& pc,
            const gtsam::Pose3& pG,
            boost::optional<gtsam::Matrix&> Hg = boost::none,
            boost::optional<gtsam::Matrix&> Ho = boost::none,
            boost::optional<gtsam::Matrix&> Hc = boost::none,
            boost::optional<gtsam::Matrix&> HG = boost::none) const {

            typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HG_, HGgo;
            typename gtsam::Matrix36 HcGgo;
            gtsam::Matrix Hy = (gtsam::Matrix13() << 0,1,0).finished();

            gtsam::Pose3 pco = gtsam::traits<gtsam::Pose3>::Between(pc,po);
            gtsam::Pose3 pgo = gtsam::traits<gtsam::Pose3>::Between(pg,po);
            gtsam::Pose3 pGgo = gtsam::traits<gtsam::Pose3>::Compose(pG,pgo,&HG_,boost::none);
            gtsam::Pose3 pcGgo = gtsam::traits<gtsam::Pose3>::Between(pc,pGgo,boost::none,&HGgo);
            gtsam::Vector c = pco.transformFrom(oc);
            gtsam::Vector C = pcGgo.transformFrom(oc,&HcGgo,boost::none);
            gtsam::Vector dc = C-c;

            if (Hg) *Hg = gtsam::Matrix16::Zero();
            if (Ho) *Ho = gtsam::Matrix16::Zero();
            if (Hc) *Hc = gtsam::Matrix16::Zero();

            if (dc[1] < 0) {
              if (HG) *HG = gtsam::Matrix16::Zero();
              return (gtsam::Vector1() << 0.0).finished();
            } else {
              if (HG) *HG = Hy * HcGgo * HGgo * HG_;
              return (gtsam::Vector1() << dc[1]).finished();
            }
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class MotionHinge

} /// namespace gtsam_custom_factors