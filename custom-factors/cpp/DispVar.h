/*
Relatess poses to displacement variable (pg, pn, po --> d)
d = CanonicalCoord( ((pn^(-1) * pg)^(-1) * (po^(-1) * pg)) )
*/

#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class DispVar: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6> {

public:

  DispVar(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6>(
        model, key1, key2, key3, key4) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& pg, const gtsam::Pose3& pn,
    const gtsam::Pose3& po, const gtsam::Vector6& d,
    boost::optional<gtsam::Matrix&> Hg = boost::none,
    boost::optional<gtsam::Matrix&> Hn = boost::none,
    boost::optional<gtsam::Matrix&> Ho = boost::none,
    boost::optional<gtsam::Matrix&> Hd = boost::none) const {

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Ho1, Hg1, Hn2, Hg2, Hng, Hog;
      gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po,pg,&Ho1,&Hg1);
      gtsam::Pose3 png = gtsam::traits<gtsam::Pose3>::Between(pn,pg,&Hn2,&Hg2);
      gtsam::Vector lm = gtsam::traits<gtsam::Pose3>::Local(png,pog,&Hng,&Hog);

      gtsam::Vector output = d - lm;

      if (Hg) *Hg = - Hog * Hg1 - Hng * Hg2;
      if (Hn) *Hn = - Hng * Hn2;
      if (Ho) *Ho = - Hog * Ho1;
      if (Hd) *Hd = gtsam::Matrix::Identity(6,6);

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class DispVar

} /// namespace gtsam_custom_factors