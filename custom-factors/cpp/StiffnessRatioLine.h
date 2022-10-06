#pragma once

#include <ostream>
#include <cmath>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

/*
Pose3: G(i)
Vector6: W(i)
Pose3: C(i)
Vector9: Compliance + Acting Point Offset (x,y,z)
*/

class StiffnessRatioLine: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector6, gtsam::Pose3, gtsam::Vector9> {

private:

  gtsam::Vector6 v_;

public:

  StiffnessRatioLine(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    const gtsam::Vector6 v_nominal, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector6, gtsam::Pose3, gtsam::Vector9>(model, key1, key2, key3, key4),
      v_(v_nominal) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pg, const gtsam::Vector6& w, const gtsam::Pose3& pc,
    const gtsam::Vector9& v,
    boost::optional<gtsam::Matrix&> Hg = boost::none,
    boost::optional<gtsam::Matrix&> Hw = boost::none,
    boost::optional<gtsam::Matrix&> Hc = boost::none,
    boost::optional<gtsam::Matrix&> Hv = boost::none) const {

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hg0, Hc0;
      gtsam::Matrix36 Hgc1;
      gtsam::Matrix33 Hgcr, Hxr, HxM;
      
      gtsam::Pose3 pgc = gtsam::traits<gtsam::Pose3>::Between(pg,pc,&Hg0,&Hc0);
      gtsam::Rot3 pgcr = pgc.rotation(&Hgc1);
      gtsam::Point3 pgct = pgc.translation();
      gtsam::Matrix Hgc2 = (gtsam::Matrix36() << 0, 0, 0, 1, 0, 0,
                                                 0, 0, 0, 0, 1, 0,
                                                 0, 0, 0, 0, 0, 1).finished();

      gtsam::Point3 pgct_ = (gtsam::Vector3() << pgct.x()-v[6], pgct.y()-v[7], pgct.z()-v[8]).finished();
      gtsam::Matrix Ht_v = (gtsam::Matrix39() << 0, 0, 0, 0, 0, 0, -1, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, -1, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, -1).finished();

      gtsam::Vector Mg = (gtsam::Vector3() << w[0], w[1], w[2]).finished();
      gtsam::Vector Fg = (gtsam::Vector3() << w[3], w[4], w[5]).finished();
      gtsam::Matrix HMgw = (gtsam::Matrix36() << 1, 0, 0, 0, 0, 0,
                                                 0, 1, 0, 0, 0, 0,
                                                 0, 0, 1, 0, 0, 0).finished();
      gtsam::Matrix HFgw = (gtsam::Matrix36() << 0, 0, 0, 1, 0, 0,
                                                 0, 0, 0, 0, 1, 0,
                                                 0, 0, 0, 0, 0, 1).finished();
      

      gtsam::Vector Mgc = (gtsam::Vector3() << 
                                Mg[0] - Fg[2]*pgct_.y() + Fg[1]*pgct_.z(),
                                Mg[1] - Fg[0]*pgct_.z() + Fg[2]*pgct_.x(),
                                Mg[2] - Fg[1]*pgct_.x() + Fg[0]*pgct_.y()
                                ).finished();
      gtsam::Matrix HMg = (gtsam::Matrix33() << 1, 0, 0,
                                                0, 1, 0,
                                                0, 0, 1).finished();
      gtsam::Matrix HFg = (gtsam::Matrix33() << 0, pgct_.z(), -pgct_.y(),
                                                -pgct_.z(), 0, pgct_.x(),
                                                pgct_.y(), -pgct_.x(), 0).finished();
      gtsam::Matrix Hgct_ = (gtsam::Matrix33() << 0, -Fg[2], Fg[1],
                                                 Fg[2], 0, -Fg[0],
                                                 -Fg[1], Fg[0], 0).finished();

      gtsam::Vector Mcx = pgcr.unrotate(Mgc, &Hxr, &HxM);

      gtsam::Vector output = (gtsam::Vector2() <<
                                Mcx[0],
                                v[0]*v[1]*v[2]*v[3]*v[4]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]) - 1
                                ).finished();
      gtsam::Matrix HMcx = (gtsam::Matrix23() << 1, 0, 0,
                                                 0, 0, 0).finished();
      gtsam::Matrix Hvv = (gtsam::Matrix29() <<
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              v[1]*v[2]*v[3]*v[4]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), v[0]*v[2]*v[3]*v[4]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), v[0]*v[1]*v[3]*v[4]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), v[0]*v[1]*v[2]*v[4]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), v[0]*v[1]*v[2]*v[3]*v[5]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), v[0]*v[1]*v[2]*v[3]*v[4]/(v_[0]*v_[1]*v_[2]*v_[3]*v_[4]*v_[5]), 0, 0, 0
                              ).finished();

      if (Hg) *Hg = (HMcx*Hxr*Hgc1 + HMcx*HxM*Hgct_*Hgc2)*Hg0;
      if (Hw) *Hw = HMcx*HxM*HMg*HMgw + HMcx*HxM*HFg*HFgw;
      if (Hc) *Hc = (HMcx*Hxr*Hgc1 + HMcx*HxM*Hgct_*Hgc2)*Hc0;
      if (Hv) *Hv = Hvv + HMcx*HxM*Hgct_*Ht_v;

      return output;

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; //

} /// namespace gtsam_packing
