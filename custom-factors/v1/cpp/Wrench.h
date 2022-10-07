#pragma once

#include <ostream>
#include <cmath>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {


/*
Pose3: G(i)
Pose3: N(i)
Pose3: O(i)
Vector6: W(i) (Wrench)
Vector9: Compliance + Acting Point Offset (x,y,z)
*/

class Wrench: public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector9> {

public:

  Wrench(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5,
    gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector9>(model, key1, key2, key3, key4, key5) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pg, const gtsam::Pose3& pn, const gtsam::Pose3& po,
    const gtsam::Vector6& w, const gtsam::Vector9& v,
    boost::optional<gtsam::Matrix&> Hg = boost::none,
    boost::optional<gtsam::Matrix&> Hn = boost::none,
    boost::optional<gtsam::Matrix&> Ho = boost::none,
    boost::optional<gtsam::Matrix&> Hw = boost::none,
    boost::optional<gtsam::Matrix&> Hv = boost::none) const {

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Ho1, Hg1, Hn2, Hg2, Hng, Hog;

      gtsam::Vector k = (gtsam::Vector6() << v[0], v[1], v[2], v[3], v[4], v[5]).finished();
      gtsam::Matrix Hk = (gtsam::Matrix69() << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 1, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 1, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 1, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 1, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 1, 0, 0, 0).finished();

      gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po,pg,&Ho1,&Hg1);
      gtsam::Pose3 png = gtsam::traits<gtsam::Pose3>::Between(pn,pg,&Hn2,&Hg2);
      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(png,pog,&Hng,&Hog);
      gtsam::Vector lm_ = (gtsam::Vector6() << lm[0], lm[1], lm[2], lm[3]+lm[1]*v[8]-lm[2]*v[7], lm[4]+lm[2]*v[6]-lm[0]*v[8], lm[5]+lm[0]*v[7]-lm[1]*v[6]).finished();
      gtsam::Vector wrench = (gtsam::Vector6() << k[0]*lm_[0], k[1]*lm_[1], k[2]*lm_[2], k[3]*lm_[3], k[4]*lm_[4], k[5]*lm_[5]).finished();
      gtsam::Vector output = w - wrench;
      
      gtsam::Matrix Hlm_v = (gtsam::Matrix69() << 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, -lm[2], lm[1],
                                                  0, 0, 0, 0, 0, 0, lm[2], 0, -lm[0],
                                                  0, 0, 0, 0, 0, 0, -lm[1], lm[0], 0).finished();
      gtsam::Matrix Hlm_lm = (gtsam::Matrix66() << 1, 0, 0, 0, 0, 0,
                                                   0, 1, 0, 0, 0, 0,
                                                   0, 0, 1, 0, 0, 0,
                                                   0, v[8], -v[7], 1, 0, 0,
                                                   -v[8], 0, v[6], 0, 1, 0,
                                                   v[7], -v[6], 0, 0, 0, 1).finished();
      gtsam::Matrix Hw_k = (gtsam::Matrix66() << lm_[0], 0, 0, 0, 0, 0,
                                                 0, lm_[1], 0, 0, 0, 0, 
                                                 0, 0, lm_[2], 0, 0, 0,
                                                 0, 0, 0, lm_[3], 0, 0,
                                                 0, 0, 0, 0, lm_[4], 0,
                                                 0, 0, 0, 0, 0, lm_[5]).finished();
      gtsam::Matrix Hw_lm_ = (gtsam::Matrix66() << k[0], 0, 0, 0, 0, 0,
                                                   0, k[1], 0, 0, 0, 0, 
                                                   0, 0, k[2], 0, 0, 0,
                                                   0, 0, 0, k[3], 0, 0,
                                                   0, 0, 0, 0, k[4], 0,
                                                   0, 0, 0, 0, 0, k[5]).finished();

      if (Hg) *Hg = - Hw_lm_ * Hlm_lm * (Hog * Hg1 + Hng * Hg2);
      if (Hn) *Hn = - Hw_lm_ * Hlm_lm * Hng * Hn2;
      if (Ho) *Ho = - Hw_lm_ * Hlm_lm * Hog * Ho1;
      if (Hw) *Hw = gtsam::Matrix::Identity(6, 6);
      if (Hv) *Hv = - (Hw_k * Hk + Hw_lm_ * Hlm_v);      

      return output;
      
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class Wrench

} /// namespace gtsam_custom_factors