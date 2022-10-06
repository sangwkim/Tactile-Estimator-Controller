#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class WrenchPredictFactor: public gtsam::NoiseModelFactor5<gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector9> {

private:

  bool zj;

public:

  WrenchPredictFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, gtsam::SharedNoiseModel model, bool zeroJac) :
      gtsam::NoiseModelFactor5<gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector9>(
        model, key1, key2, key3, key4, key5), zj(zeroJac) {}

  gtsam::Vector evaluateError(const gtsam::Vector6& d_, const gtsam::Vector6& d,
    const gtsam::Vector6& w_, const gtsam::Vector6& w, const gtsam::Vector9& v,
    boost::optional<gtsam::Matrix&> Hd_ = boost::none,
    boost::optional<gtsam::Matrix&> Hd = boost::none,
    boost::optional<gtsam::Matrix&> Hw_ = boost::none,
    boost::optional<gtsam::Matrix&> Hw = boost::none,
    boost::optional<gtsam::Matrix&> Hv = boost::none) const {

      gtsam::Vector k = (gtsam::Vector6() << v[0], v[1], v[2], v[3], v[4], v[5]).finished();
      gtsam::Vector dc = (gtsam::Vector6() << d[0], d[1], d[2], d[3]+d[1]*v[8]-d[2]*v[7], d[4]+d[2]*v[6]-d[0]*v[8], d[5]+d[0]*v[7]-d[1]*v[6]).finished();
      gtsam::Vector dc_ = (gtsam::Vector6() << d_[0], d_[1], d_[2], d_[3]+d_[1]*v[8]-d_[2]*v[7], d_[4]+d_[2]*v[6]-d_[0]*v[8], d_[5]+d_[0]*v[7]-d_[1]*v[6]).finished();
      gtsam::Vector diff = dc - dc_;
      gtsam::Vector kdiff = (gtsam::Vector6() << k[0]*diff[0], k[1]*diff[1], k[2]*diff[2], k[3]*diff[3], k[4]*diff[4], k[5]*diff[5]).finished();

      gtsam::Matrix Hdcd = (gtsam::Matrix66() << 1, 0, 0, 0, 0, 0,
                                                 0, 1, 0, 0, 0, 0,
                                                 0, 0, 1, 0, 0, 0,
                                                 0, v[8], -v[7], 1, 0, 0,
                                                 -v[8], 0, v[6], 0, 1, 0,
                                                 v[7], -v[6], 0, 0, 0, 1).finished();
      gtsam::Matrix Hkdiff = k.asDiagonal();

      gtsam::Vector output = w - w_ - kdiff;

      if (zj==true) {
        if (Hd_) *Hd_ = gtsam::Matrix::Zero(6,6);
        if (Hw_) *Hw_ = gtsam::Matrix::Zero(6,6);
      } else {
        if (Hd_) *Hd_ = Hkdiff * Hdcd;
        if (Hw_) *Hw_ = - gtsam::Matrix::Identity(6,6);
      }
      if (Hd) *Hd = - Hkdiff * Hdcd;
      if (Hw) *Hw = gtsam::Matrix::Identity(6,6);
      if (Hv) *Hv = gtsam::Matrix::Zero(6,9);

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class WrenchPredictFactor

} /// namespace gtsam_packing