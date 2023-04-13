#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class PhysicInfer: public gtsam::NoiseModelFactor6<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6> {

private:

    gtsam::Point3 oc;

public:

  PhysicInfer(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, gtsam::Key key6, const gtsam::Point3 contact_point, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor6<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector6>(model, key1, key2, key3, key4, key5, key6),
      oc(contact_point) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& po,
            const gtsam::Pose3& pO,
            const gtsam::Pose3& pg,
            const gtsam::Pose3& pG,
            const gtsam::Vector3& s,
            const gtsam::Vector6& w,
            boost::optional<gtsam::Matrix&> Ho = boost::none,
            boost::optional<gtsam::Matrix&> HO = boost::none,
            boost::optional<gtsam::Matrix&> Hg = boost::none,
            boost::optional<gtsam::Matrix&> HG = boost::none,
            boost::optional<gtsam::Matrix&> Hs = boost::none,
            boost::optional<gtsam::Matrix&> Hw = boost::none) const {
      
        typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Ho_, Hg_, HO_, HG_, Hog, HOG;
        typename gtsam::Matrix36 Ho__, HO__;
        typename gtsam::Matrix33 Hdum1, Hdum2;

        // Parameter Conversion
        gtsam::Vector e = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished();
          // Jacobian
        gtsam::Matrix Hes = (gtsam::Vector3() << exp(s[0]), exp(s[1]), exp(s[2])).finished().asDiagonal();
        
        // Intrinsic Transformation
        gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po,pg,&Ho_,&Hg_);
        gtsam::Pose3 pOG = gtsam::traits<gtsam::Pose3>::Between(pO,pG,&HO_,&HG_);
        gtsam::Vector lmi = gtsam::traits<gtsam::Pose3>::Local(pog,pOG,&Hog,&HOG);
          // Potential due to intrinsic sliding (psi_i)
        double psi_i = lmi[2]*lmi[2]/e[0]/e[0] + lmi[3]*lmi[3]/e[1]/e[1] + lmi[4]*lmi[4]/e[1]/e[1];
          // Jacobians
            // d( Intrinsic Potential ) / d( Object Poses )
        gtsam::Matrix Hpsii_lmi = (gtsam::Matrix16() << 0, 0, 2*lmi[2]/e[0]/e[0], 2*lmi[3]/e[1]/e[1], 2*lmi[4]/e[1]/e[1], 0).finished();
        gtsam::Matrix Hpsiio = Hpsii_lmi * Hog * Ho_;
        gtsam::Matrix HpsiiO = Hpsii_lmi * HOG * HO_;
            // Jacobian of Jacobian
        gtsam::Matrix H_Hpsii_lmi_e = (gtsam::Matrix63() << 0, 0, 0,
                                                            0, 0, 0,
                                                            -4*lmi[2]/e[0]/e[0]/e[0], 0, 0,
                                                            0, -4*lmi[3]/e[1]/e[1]/e[1], 0,
                                                            0, -4*lmi[4]/e[1]/e[1]/e[1], 0,
                                                            0, 0, 0).finished();
              // d( d( Intrinsic Potential ) / d( Object Poses ) ) / d( Parameter )
        gtsam::Matrix H_Hpsiio_e = (H_Hpsii_lmi_e.transpose() * Hog * Ho_).transpose();
        gtsam::Matrix H_HpsiiO_e = (H_Hpsii_lmi_e.transpose() * HOG * HO_).transpose();

        // Extrinsic Transformation
        gtsam::Vector c = po.transformFrom(oc, &Ho__, &Hdum1);
        gtsam::Vector C = pO.transformFrom(oc, &HO__, &Hdum2);
        gtsam::Vector cC = c - C;
          // Potential due to extrinsic sliding (psi_e)
        double psi_e = std::abs(w[4])*(cC[0]*cC[0] + cC[2]*cC[2])/e[2]/e[2];
          // Jacobians
            // d( Extrinsic Potential ) / d( Object Poses )
        gtsam::Matrix Hpsie_cC = std::abs(w[4])*(gtsam::Matrix13() << 2*cC[0]/e[2]/e[2], 0, 2*cC[2]/e[2]/e[2]).finished();
        gtsam::Matrix Hpsieo = Hpsie_cC * Ho__;
        gtsam::Matrix HpsieO = - Hpsie_cC * HO__;
            // Jacobian of Jacobian
        gtsam::Matrix H_Hpsie_cC_e = std::abs(w[4])*(gtsam::Matrix33() << 0, 0, -4*cC[0]/e[2]/e[2]/e[2],
                                                           0, 0, 0,
                                                           0, 0, -4*cC[2]/e[2]/e[2]/e[2]).finished();
              // d( d( Extrinsic Potential ) / d( Object Poses ) ) / d( Parameters )
        gtsam::Matrix H_Hpsieo_e = (H_Hpsie_cC_e.transpose() * Ho__).transpose();
        gtsam::Matrix H_HpsieO_e = (- H_Hpsie_cC_e.transpose() * HO__).transpose();

        // Total Potential
        double psi = psi_i + psi_e;
          // Jacobian
        gtsam::Matrix Hpsio = Hpsiio + Hpsieo;
        gtsam::Matrix HpsiO = HpsiiO + HpsieO;

        // Normal Component (y-direction)
        double y = c[1];
        double Y = C[1];
          // Jacobian
        gtsam::Matrix Hyo = (gtsam::Matrix13() << 0, 1, 0).finished() * Ho__;
        gtsam::Matrix HYO = (gtsam::Matrix13() << 0, 1, 0).finished() * HO__;

        // Output
        gtsam::Vector output = (gtsam::Vector(10) << Hpsio(0,0)*Hyo(0,4) - Hpsio(0,4)*Hyo(0,0),
                                                     Hpsio(0,1)*Hyo(0,4) - Hpsio(0,4)*Hyo(0,1),
                                                     Hpsio(0,2)*Hyo(0,4) - Hpsio(0,4)*Hyo(0,2),
                                                     Hpsio(0,3)*Hyo(0,4) - Hpsio(0,4)*Hyo(0,3),
                                                     Hpsio(0,5)*Hyo(0,4) - Hpsio(0,4)*Hyo(0,5),
                                                     HpsiO(0,0)*HYO(0,4) - HpsiO(0,4)*HYO(0,0),
                                                     HpsiO(0,1)*HYO(0,4) - HpsiO(0,4)*HYO(0,1),
                                                     HpsiO(0,2)*HYO(0,4) - HpsiO(0,4)*HYO(0,2),
                                                     HpsiO(0,3)*HYO(0,4) - HpsiO(0,4)*HYO(0,3),
                                                     HpsiO(0,5)*HYO(0,4) - HpsiO(0,4)*HYO(0,5)
                                                     ).finished();
        
        gtsam::Matrix H_out_Hpsio = (gtsam::Matrix(10,6) << Hyo(0,4), 0, 0, 0, -Hyo(0,0), 0,
                                                            0, Hyo(0,4), 0, 0, -Hyo(0,1), 0,
                                                            0, 0, Hyo(0,4), 0, -Hyo(0,2), 0,
                                                            0, 0, 0, Hyo(0,4), -Hyo(0,3), 0,
                                                            0, 0, 0, 0, -Hyo(0,5), Hyo(0,4),
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0
                                                            ).finished();
        
        gtsam::Matrix H_out_HpsiO = (gtsam::Matrix(10,6) << 0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            HYO(0,4), 0, 0, 0, -HYO(0,0), 0,
                                                            0, HYO(0,4), 0, 0, -HYO(0,1), 0,
                                                            0, 0, HYO(0,4), 0, -HYO(0,2), 0,
                                                            0, 0, 0, HYO(0,4), -HYO(0,3), 0,
                                                            0, 0, 0, 0, -HYO(0,5), HYO(0,4)).finished();
        

        gtsam::Matrix H_out_e = H_out_Hpsio * (H_Hpsiio_e + H_Hpsieo_e) + H_out_HpsiO * (H_HpsiiO_e + H_HpsieO_e);
        //gtsam::Matrix H_out_e = H_out_HpsiO * (H_HpsiiO_e + H_HpsieO_e);

        if (Ho) *Ho = gtsam::Matrix::Zero(10,6);
        if (HO) *HO = gtsam::Matrix::Zero(10,6);
        if (Hg) *Hg = gtsam::Matrix::Zero(10,6);
        if (HG) *HG = gtsam::Matrix::Zero(10,6);
        if (Hs) *Hs = H_out_e * Hes;
        if (Hw) *Hw = gtsam::Matrix::Zero(10,6);
        /*
        std::cout << "e: " << e << std::endl;
        std::cout << "lmi: " << lmi << std::endl;
        std::cout << "Hpsii_lmi: " << Hpsii_lmi << std::endl;
        std::cout << "Hpsiio: " << Hpsiio << std::endl;
        std::cout << "Hpsieo: " << Hpsieo << std::endl;
        std::cout << "HpsiiO: " << HpsiiO << std::endl;
        std::cout << "HpsieO: " << HpsieO << std::endl;
        std::cout << "Hpsio: " << Hpsio << std::endl;
        std::cout << "HpsiO: " << HpsiO << std::endl;
        std::cout << "Hyo: " << Hyo << std::endl;
        std::cout << "HYO: " << HYO << std::endl;
        std::cout << "output: " << output << std::endl;
        std::cout << "output_1: " << (gtsam::Vector(10) << Hpsio(0,0)*Hyo(0,4),
                                                     Hpsio(0,1)*Hyo(0,4),
                                                     Hpsio(0,2)*Hyo(0,4),
                                                     Hpsio(0,3)*Hyo(0,4),
                                                     Hpsio(0,5)*Hyo(0,4),
                                                     HpsiO(0,0)*HYO(0,4),
                                                     HpsiO(0,1)*HYO(0,4),
                                                     HpsiO(0,2)*HYO(0,4),
                                                     HpsiO(0,3)*HYO(0,4),
                                                     HpsiO(0,5)*HYO(0,4)
                                                     ).finished() << std::endl;
        std::cout << "output_2: " << (gtsam::Vector(10) << Hpsio(0,4)*Hyo(0,0),
                                                     Hpsio(0,4)*Hyo(0,1),
                                                     Hpsio(0,4)*Hyo(0,2),
                                                     Hpsio(0,4)*Hyo(0,3),
                                                     Hpsio(0,4)*Hyo(0,5),
                                                     HpsiO(0,4)*HYO(0,0),
                                                     HpsiO(0,4)*HYO(0,1),
                                                     HpsiO(0,4)*HYO(0,2),
                                                     HpsiO(0,4)*HYO(0,3),
                                                     HpsiO(0,4)*HYO(0,5)
                                                     ).finished() << std::endl;
        std::cout << "Hs: " << *Hs << std::endl;
        std::cout << "H_out_e: " << H_out_e << std::endl;
        std::cout << "H_out_Hpsio * (H_Hpsiio_e + H_Hpsieo_e): " << H_out_Hpsio * (H_Hpsiio_e + H_Hpsieo_e) << std::endl;
        std::cout << "H_out_HpsiO * (H_HpsiiO_e + H_HpsieO_e): " << H_out_HpsiO * (H_HpsiiO_e + H_HpsieO_e) << std::endl;        
        std::cout << "H_out_Hpsio: " << H_out_Hpsio << std::endl;
        std::cout << "H_out_HpsiO: " << H_out_HpsiO << std::endl;
        std::cout << "H_Hpsiio_e: " << H_Hpsiio_e << std::endl;
        std::cout << "H_Hpsieo_e: " << H_Hpsieo_e << std::endl;
        std::cout << "H_HpsiiO_e: " << H_HpsiiO_e << std::endl;
        std::cout << "H_HpsieO_e: " << H_HpsieO_e << std::endl;
        */
        return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 6;
  }

}; // \class PhysicInfer

} /// namespace gtsam_custom_factors