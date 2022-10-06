#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class PenetrationFactor_2: public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

  double epsilon_;

public:

  PenetrationFactor_2(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, double cost_sigma, double eps) :
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(1, cost_sigma), key1, key2, key3),
      epsilon_(eps) {}

  gtsam::Vector evaluateError(
    const gtsam::Pose3& pn,
    const gtsam::Pose3& pc,
    const gtsam::Pose3& po,
    boost::optional<gtsam::Matrix&> Hn = boost::none,
    boost::optional<gtsam::Matrix&> Hc = boost::none,
    boost::optional<gtsam::Matrix&> Ho = boost::none
    ) const {

      gtsam::Pose3 poc = gtsam::traits<gtsam::Pose3>::Between(po, pc, Ho, Hc);

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hoc;
      gtsam::Pose3 pnoc = gtsam::traits<gtsam::Pose3>::Compose(pn, poc, Hn, &Hoc);

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hc_;
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hnoc;
      gtsam::Pose3 pcnoc = gtsam::traits<gtsam::Pose3>::Between(pc, pnoc, &Hc_, &Hnoc);

      double pcnocz = - pcnoc.translation().z();
      gtsam::Matrix16 Hz = (gtsam::Matrix16() << 0, 0, 0, 0, 0, -1).finished();

      if (pcnocz > epsilon_) {
        if (Hn) *Hn = gtsam::Matrix16::Zero();
        if (Hc) *Hc = gtsam::Matrix16::Zero();
        if (Ho) *Ho = gtsam::Matrix16::Zero();
        return (gtsam::Vector1() << 0.0).finished();
      } else {
        if (Hn) *Hn = - Hz * Hnoc * (*Hn);
        if (Hc) *Hc = - (Hz * Hnoc * Hoc * (*Hc) + Hz * Hc_);
        if (Ho) *Ho = - Hz * Hnoc * Hoc * (*Ho);
        return (gtsam::Vector1() << epsilon_ - pcnocz).finished();
      }
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 3;
  }

}; // \class PenetrationFactor_2

} /// namespace gtsam_packing