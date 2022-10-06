#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

// Computes C^-1 * N * O^-1 * C
//            = (N^-1 * C)^-1 * (O^-1 * C)

namespace gtsam_packing {

class NOCFactor: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

public:

  NOCFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3,
    gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pn, const gtsam::Pose3& po, const gtsam::Pose3& pc,
    boost::optional<gtsam::Matrix&> Hn = boost::none,
    boost::optional<gtsam::Matrix&> Ho = boost::none,
    boost::optional<gtsam::Matrix&> Hc = boost::none) const {
      
      //typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hoc, Hc_, Hnoc;
      //gtsam::Pose3 poc = gtsam::traits<gtsam::Pose3>::Between(po,pc,Ho,Hc);
      //gtsam::Pose3 pnoc = gtsam::traits<gtsam::Pose3>::Compose(pn,poc,Hn,&Hoc);
      //gtsam::Vector cnoc = gtsam::traits<gtsam::Pose3>::Local(pc,pnoc,&Hc_,&Hnoc);

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hc_, Hnc, Hoc;
      gtsam::Pose3 poc = gtsam::traits<gtsam::Pose3>::Between(po,pc,Ho,Hc);
      gtsam::Pose3 pnc = gtsam::traits<gtsam::Pose3>::Between(pn,pc,Hn,&Hc_);
      gtsam::Vector cnoc = gtsam::traits<gtsam::Pose3>::Local(pnc,poc,&Hnc,&Hoc);

      //if (Hn) *Hn = Hnoc * (*Hn);
      //if (Ho) *Ho = Hnoc * Hoc * (*Ho);
      //if (Hc) *Hc = Hc_ + Hnoc * Hoc * (*Hc);

      if (Hn) *Hn = Hnc * (*Hn);
      if (Ho) *Ho = Hoc * (*Ho);
      if (Hc) *Hc = Hnc * Hc_ + Hoc * (*Hc);

      return cnoc;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class NOCFactor

} /// namespace gtsam_packing