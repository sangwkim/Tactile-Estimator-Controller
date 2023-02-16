#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

/*
G(i), N(i), O(i), C(i), C(i-1)
*/

class Friction: public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

  double mu_;
  gtsam::Vector6 k_;
  double s_weak;
  double s_strong;

public:

  Friction(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, double mu, const gtsam::Vector6 k, double sigma_weak, double sigma_strong) :
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(1,1.), key1, key2, key3, key4, key5),
        mu_(mu), k_(k), s_weak(sigma_weak), s_strong(sigma_strong) {}

  gtsam::Vector evaluateError(
    const gtsam::Pose3& pg, const gtsam::Pose3& pn, const gtsam::Pose3& po, const gtsam::Pose3& pc, const gtsam::Pose3& pc_,
    boost::optional<gtsam::Matrix&> Hg = boost::none,
    boost::optional<gtsam::Matrix&> Hn = boost::none,
    boost::optional<gtsam::Matrix&> Ho = boost::none,
    boost::optional<gtsam::Matrix&> Hc = boost::none,
    boost::optional<gtsam::Matrix&> Hc_ = boost::none) const {

      // Compute tactile displacement and force (gripper frame)
      gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po,pg);
      gtsam::Pose3 png = gtsam::traits<gtsam::Pose3>::Between(pn,pg);
      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(png,pog); // tactile displacement
      gtsam::Vector Fg = (gtsam::Vector3() << k_[3]*lm[3], k_[4]*lm[4], k_[5]*lm[5]).finished(); // tactile force (gripper frame)
      
      // Convert the tactile force from gripper frame to world (contact) frame
      gtsam::Pose3 pgc = gtsam::traits<gtsam::Pose3>::Between(pg,pc);
      gtsam::Rot3 pgcr = pgc.rotation();
      gtsam::Vector Fc = pgcr.unrotate(Fg);

      // Compute normal and tangential force component
      double FN = - Fc[2];
      double FT = gtsam::norm3((gtsam::Vector3() << Fc[0], Fc[1], 0).finished());

      // Friction Cone
      double val = FT - mu_ * FN;

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hc1, Hc1_;
      gtsam::Vector pcc = gtsam::traits<gtsam::Pose3>::Local(pc,pc_,&Hc1,&Hc1_);
      gtsam::Vector pcct = (gtsam::Vector3() << pcc[3], pcc[4], pcc[5]).finished();
      gtsam::Matrix Hpcc = (gtsam::Matrix36() << 0, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 0, 1, 0,
                                                  0, 0, 0, 0, 0, 1).finished();
      typename gtsam::Matrix13 Hpcct;
      double slip = gtsam::norm3((gtsam::Vector3() << pcct[0], pcct[1], 0).finished(), &Hpcct);

      if ( val > 0 ) {
          if (Hg) *Hg = gtsam::Matrix16::Zero();
          if (Hn) *Hn = gtsam::Matrix16::Zero();
          if (Ho) *Ho = gtsam::Matrix16::Zero();
          if (Hc) *Hc = Hpcct * Hpcc * Hc1 / s_weak;
          if (Hc_) *Hc_ = Hpcct * Hpcc * Hc1_ / s_weak;
          return (gtsam::Vector1() << slip / s_weak).finished();
      } else {
          if (Hg) *Hg = gtsam::Matrix16::Zero();
          if (Hn) *Hn = gtsam::Matrix16::Zero();
          if (Ho) *Ho = gtsam::Matrix16::Zero();
          if (Hc) *Hc = Hpcct * Hpcc * Hc1 / s_strong;
          if (Hc_) *Hc_ = Hpcct * Hpcc * Hc1_ / s_strong;
          return (gtsam::Vector1() << slip / s_strong).finished();
      }
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class Friction

} /// namespace gtsam_custom_factors