#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class WrenchEnergyFactor: public gtsam::NoiseModelFactor2<gtsam::Vector6, gtsam::Vector9> {

public:

  WrenchEnergyFactor(gtsam::Key key1, gtsam::Key key2, double cost_sigma) :
      gtsam::NoiseModelFactor2<gtsam::Vector6, gtsam::Vector9>(
        gtsam::noiseModel::Isotropic::Sigma(6,cost_sigma), key1, key2) {}

  gtsam::Vector evaluateError(const gtsam::Vector6& w, const gtsam::Vector9& v,
    boost::optional<gtsam::Matrix&> Hw = boost::none,
    boost::optional<gtsam::Matrix&> Hv = boost::none) const {

      gtsam::Vector output = (gtsam::Vector6() << pow(v[0],-0.5)*w[0], pow(v[1],-0.5)*w[1], pow(v[2],-0.5)*w[2], pow(v[3],-0.5)*w[3], pow(v[4],-0.5)*w[4], pow(v[5],-0.5)*w[5]).finished();

      if (Hw) *Hw = (gtsam::Vector6() << pow(v[0],-0.5), pow(v[1],-0.5), pow(v[2],-0.5), pow(v[3],-0.5), pow(v[4],-0.5), pow(v[5],-0.5)).finished().asDiagonal();
      if (Hv) *Hv = gtsam::Matrix::Zero(6,9);

      return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class WrenchEnergyFactor

} /// namespace gtsam_packing