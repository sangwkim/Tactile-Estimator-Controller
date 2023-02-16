#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

/*
G(i), N(i), O(i), C(i), C(i-1)
*/

class FrictionLine: public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

  double mu_;
  gtsam::Vector6 k_;
  double s_weak;
  double s_strong;
  double l_;

public:

  FrictionLine(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, double mu, const gtsam::Vector6 k, double sigma_weak, double sigma_strong, double l) :
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(gtsam::noiseModel::Isotropic::Sigma(2,1.), key1, key2, key3, key4, key5),
        mu_(mu), k_(k), s_weak(sigma_weak), s_strong(sigma_strong), l_(l) {}

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
      gtsam::Vector Mg = (gtsam::Vector3() << k_[0]*lm[0], k_[1]*lm[1], k_[2]*lm[2]).finished(); // tactile torque (gripper frame)
      
      // Convert the tactile force from gripper frame to world (contact) frame
      gtsam::Pose3 pcg = gtsam::traits<gtsam::Pose3>::Between(pc,pg);
      gtsam::Rot3 pcgr = pcg.rotation();
      gtsam::Vector Fg_ = pcgr.rotate(Fg); // gripper force in contact frame
      double ratio = Fg_[0] / Fg_[2]; // ratio: along contact line <-->  along contact normal
      gtsam::Vector Mg_ = pcgr.rotate(Mg); // gripper torque in contact frame
      gtsam::Vector r1_ = - pcg.translation(); // TCP to first contact point vector in contact frame
      gtsam::Vector r12_ = (gtsam::Vector3() << l_, 0, 0).finished(); // Vector from first to second contact point in contact frame
      gtsam::Vector r2_ = r1_ + r12_; // TCP to second contact point vector in contact frame

      gtsam::Vector F1_perpend = gtsam::cross(Mg_ - gtsam::cross(r2_, Fg_), r12_) / std::pow(l_,2);
      gtsam::Vector F1_paral = ratio * F1_perpend[2] * r12_  / l_;
      gtsam::Vector F1 = F1_perpend + F1_paral; // Contact Force at the first contact point (contact frame)

      gtsam::Vector F2_perpend = gtsam::cross(Mg_ - gtsam::cross(r1_, Fg_), -r12_) / std::pow(l_,2);
      gtsam::Vector F2_paral = - ratio * F2_perpend[2] * r12_  / l_;
      gtsam::Vector F2 = F2_perpend + F2_paral; // Contact Force at the second contact point (contact frame)

      // Compute normal and tangential force component
      double FN1 = F1[2];
      double FT1 = gtsam::norm3((gtsam::Vector3() << F1[0], F1[1], 0).finished());
      double FN2 = F2[2];
      double FT2 = gtsam::norm3((gtsam::Vector3() << F2[0], F2[1], 0).finished());

      // FrictionLine Cone
      double val1 = FT1 - mu_ * FN1;
      double val2 = FT2 - mu_ * FN2;

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hc1, Hc1_, Hcc;
      gtsam::Pose3 pcc = gtsam::traits<gtsam::Pose3>::Between(pc,pc_,&Hc1,&Hc1_);
      gtsam::Vector pcclm = gtsam::traits<gtsam::Pose3>::Logmap(pcc, &Hcc);
      gtsam::Vector pcct = (gtsam::Vector3() << pcclm[3], pcclm[4], pcclm[5]).finished();
      gtsam::Matrix Hpcclm = (gtsam::Matrix36() << 0, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 0, 1, 0,
                                                  0, 0, 0, 0, 0, 1).finished();
      typename gtsam::Matrix13 Hpcct;
      double slip1 = gtsam::norm3((gtsam::Vector3() << pcct[0], pcct[1], 0).finished(), &Hpcct);

      typename gtsam::Matrix36 Hpcc;
      typename gtsam::Matrix33 Hdum;
      gtsam::Vector pc2 = pcc.transformFrom(r12_, &Hpcc, &Hdum);
      gtsam::Vector pcct2 = pc2 - r12_;
      typename gtsam::Matrix13 Hpcct2;
      double slip2 = gtsam::norm3((gtsam::Vector3() << pcct2[0], pcct2[1], 0).finished(), &Hpcct2);

      double cost1, cost2;
      gtsam::Matrix Hslip1, Hslip2;
      if ( val1 > 0 ) {
          cost1 = slip1 / s_weak;
          Hslip1 = (gtsam::Matrix21() << 1/s_weak, 0).finished();
      } else {
          cost1 = slip1 / s_strong;
          Hslip1 = (gtsam::Matrix21() << 1/s_strong, 0).finished();
      }
      if ( val2 > 0 ) {
          cost2 = slip2 / s_weak;
          Hslip2 = (gtsam::Matrix21() << 0, 1/s_weak).finished();
      } else {
          cost2 = slip2 / s_strong;
          Hslip2 = (gtsam::Matrix21() << 0, 1/s_strong).finished();
      }
      
      if (Hg) *Hg = gtsam::Matrix26::Zero();
      if (Hn) *Hn = gtsam::Matrix26::Zero();
      if (Ho) *Ho = gtsam::Matrix26::Zero();
      if (Hc) *Hc = Hslip1 * Hpcct * Hpcclm * Hcc * Hc1 + Hslip2 * Hpcct2 * Hpcc * Hc1;
      if (Hc_) *Hc_ = Hslip1 * Hpcct * Hpcclm * Hcc * Hc1_ + Hslip2 * Hpcct2 * Hpcc * Hc1_;

      return (gtsam::Vector2() << cost1, cost2).finished();
      
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class FrictionLine

} /// namespace gtsam_custom_factors