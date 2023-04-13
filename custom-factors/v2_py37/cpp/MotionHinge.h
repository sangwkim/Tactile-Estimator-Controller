#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class MotionHinge: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

  gtsam::Matrix33 cov;

public:

  MotionHinge(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Matrix33 c_cov, double cost_sigma) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(1, cost_sigma), key1, key2, key3),
      cov(c_cov) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& pg,
            const gtsam::Pose3& pc,
            const gtsam::Pose3& pG,
            boost::optional<gtsam::Matrix&> Hg = boost::none,
            boost::optional<gtsam::Matrix&> Hc = boost::none,
            boost::optional<gtsam::Matrix&> HG = boost::none) const {

            gtsam::Pose3 pcg = gtsam::traits<gtsam::Pose3>::Between(pc,pg);
            gtsam::Rot3 pcgr = pcg.rotation();
            gtsam::Pose3 pgG = gtsam::traits<gtsam::Pose3>::Between(pg,pG);
            gtsam::Rot3 pgGr = pgG.rotation();
            gtsam::Vector pgGrv = gtsam::traits<gtsam::Rot3>::Logmap(pgGr);
            gtsam::Vector pgGrv_ = pcgr.rotate(pgGrv);
            gtsam::Matrix cross = (gtsam::Matrix31() << pgGrv_[2], 0, -pgGrv_[0]).finished();
            gtsam::Matrix Hcross = (gtsam::Matrix33() << 0, 0, 1,
                                                         0, 0, 0,
                                                         -1, 0, 0).finished();
            gtsam::Matrix cross_T = (gtsam::Matrix13() << pgGrv_[2], 0, -pgGrv_[0]).finished();
            gtsam::Matrix var = cross_T * cov * cross;
            double epsilon = 1 * pow(var(0,0), 0.5);

            typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HG_, HGgc;
            typename gtsam::Matrix36 HcGgc;
            gtsam::Matrix Hy = (gtsam::Matrix13() << 0,1,0).finished();

            gtsam::Pose3 pgc = gtsam::traits<gtsam::Pose3>::Between(pg,pc);
            gtsam::Pose3 pGgc = gtsam::traits<gtsam::Pose3>::Compose(pG,pgc,&HG_,boost::none);
            gtsam::Pose3 pcGgc = gtsam::traits<gtsam::Pose3>::Between(pc,pGgc,boost::none,&HGgc);
            gtsam::Vector dc = pcGgc.translation(&HcGgc);

            if (Hg) *Hg = gtsam::Matrix16::Zero();
            if (Hc) *Hc = gtsam::Matrix16::Zero();

            if (dc[1] <= -epsilon) {
              if (HG) *HG = gtsam::Matrix16::Zero();
              return (gtsam::Vector1() << 0.0).finished();
            } else {
              if (HG) *HG = Hy * HcGgc * HGgc * HG_;
              return (gtsam::Vector1() << dc[1]+epsilon).finished();
            }

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class MotionHinge

} /// namespace gtsam_custom_factors