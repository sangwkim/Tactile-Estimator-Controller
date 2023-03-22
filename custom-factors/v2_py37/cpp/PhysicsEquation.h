#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_custom_factors {

class PhysicsEquation: public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

    double F_max, M_max, mu_max;

public:

  PhysicsEquation(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4, gtsam::Key key5, const double f_, const double m_, const double mu_, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3, key4, key5),
      F_max(f_), M_max(m_), mu_max(mu_) {}

  gtsam::Vector evaluateError(
            const gtsam::Pose3& po,
            const gtsam::Pose3& pO,
            const gtsam::Pose3& pg,
            const gtsam::Pose3& pG,
            const gtsam::Pose3& pc,
            boost::optional<gtsam::Matrix&> Ho = boost::none,
            boost::optional<gtsam::Matrix&> HO = boost::none,
            boost::optional<gtsam::Matrix&> Hg = boost::none,
            boost::optional<gtsam::Matrix&> HG = boost::none,
            boost::optional<gtsam::Matrix&> Hc = boost::none) const {

        typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hc1, Hg3, Hg2, HG2, Ho1, Hg1, HO1, HG1, Hog, HOG;
        typename gtsam::Matrix36 Hcg1, Hcg2;
        typename gtsam::Matrix33 Hcgr2, Hgrg, Hcgr4, Hgtg, Hcgr1, Hsrg, Hcgr3, Hstg;

        // Rotation for frame conversion
        gtsam::Pose3 pcg = gtsam::traits<gtsam::Pose3>::Between(pc,pg,&Hc1,&Hg3);
        gtsam::Rot3 pcgr = pcg.rotation(&Hcg1);
        gtsam::Vector cg_trn = pcg.translation(&Hcg2);

        // Basic Jacobian
        gtsam::Matrix Hrot = (gtsam::Matrix36() << 1, 0, 0, 0, 0, 0,
                                                   0, 1, 0, 0, 0, 0,
                                                   0, 0, 1, 0, 0, 0).finished();
        gtsam::Matrix Htrn = (gtsam::Matrix36() << 0, 0, 0, 1, 0, 0,
                                                   0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 0, 1).finished();
        
        // Gripper motion
        gtsam::Vector gG_g = gtsam::traits<gtsam::Pose3>::Local(pg,pG,&Hg2,&HG2); // in gripper frame
        gtsam::Vector gG_rot_g = (gtsam::Vector3() << gG_g[0], gG_g[1], gG_g[2]).finished(); // rotational component
        gtsam::Vector gG_trn_g = (gtsam::Vector3() << gG_g[3], gG_g[4], gG_g[5]).finished(); // translational componenet
        gtsam::Vector gG_rot = pcgr.rotate(gG_rot_g,&Hcgr2,&Hgrg);
        gtsam::Vector gG_trn = pcgr.rotate(gG_trn_g,&Hcgr4,&Hgtg);

        double XX = gG_trn[0] + (cg_trn[1] * gG_rot[2]);
        double YY = gG_trn[1] - (cg_trn[0] * gG_rot[2]);

        gtsam::Matrix Hcgt3 = (gtsam::Matrix13() << 0, gG_rot[2], 0).finished();
        gtsam::Matrix Hgr1 = (gtsam::Matrix13() << 0, 0, cg_trn[1]).finished();
        gtsam::Matrix Hgt1 = (gtsam::Matrix13() << 1, 0, 0).finished();
        gtsam::Matrix Hcgt4 = (gtsam::Matrix13() << -gG_rot[2], 0, 0).finished();
        gtsam::Matrix Hgr2 = (gtsam::Matrix13() << 0, 0, -cg_trn[0]).finished();
        gtsam::Matrix Hgt2 = (gtsam::Matrix13() << 0, 1, 0).finished();
        
        // Slip
        gtsam::Pose3 pog = gtsam::traits<gtsam::Pose3>::Between(po,pg,&Ho1,&Hg1);
        gtsam::Pose3 pOG = gtsam::traits<gtsam::Pose3>::Between(pO,pG,&HO1,&HG1);
        gtsam::Vector slip_g = gtsam::traits<gtsam::Pose3>::Local(pog,pOG,&Hog,&HOG); // slip in gripper frame
        gtsam::Vector slip_rot_g = (gtsam::Vector3() << slip_g[0], slip_g[1], slip_g[2]).finished(); // rotational component
        gtsam::Vector slip_trn_g = (gtsam::Vector3() << slip_g[3], slip_g[4], slip_g[5]).finished(); // translational componenet
        gtsam::Vector slip_rot = pcgr.rotate(slip_rot_g,&Hcgr1,&Hsrg);
        gtsam::Vector slip_trn = pcgr.rotate(slip_trn_g,&Hcgr3,&Hstg);

        // Physical Equation

        double AT = (1 / pow(F_max,2)) + (pow(cg_trn[0],2) / pow(M_max,2));
        double AN = (1 / pow(F_max,2)) + (pow(cg_trn[1],2) / pow(M_max,2));
        double B = cg_trn[0] * cg_trn[1] / pow(M_max,2);
        gtsam::Matrix Hcgt5 = (gtsam::Matrix13() << 2*cg_trn[0]/pow(M_max,2), 0, 0).finished();
        gtsam::Matrix Hcgt6 = (gtsam::Matrix13() << 0, 2*cg_trn[1]/pow(M_max,2), 0).finished();
        gtsam::Matrix Hcgt7 = (gtsam::Matrix13() << cg_trn[1]/pow(M_max,2), cg_trn[0]/pow(M_max,2), 0).finished();

        double mu = ((XX*AT)+(YY*B)) / ((XX*B)+(YY*AN));

        double x_slip, y_slip;
        gtsam::Matrix HXX1, HYY1, HAT1, HAN1, HB1, HXX2, HYY2, HAT2, HAN2, HB2;

        if (mu >= -mu_max && mu <= mu_max) {
            x_slip = (YY*B + XX*AT) / (pow(B,2) - AN*AT) / pow(F_max,2);
            y_slip = (YY*AN + XX*B) / (pow(B,2) - AN*AT) / pow(F_max,2);
            
            HXX1 = (gtsam::Matrix11() << AT / (pow(B,2) - AN*AT) / pow(F_max,2)).finished();
            HYY1 = (gtsam::Matrix11() << B / (pow(B,2) - AN*AT) / pow(F_max,2)).finished();
            HAT1 = (gtsam::Matrix11() << (XX*B*B + YY*B*AN) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();
            HAN1 = (gtsam::Matrix11() << (YY*B*AT + XX*AT*AT) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();
            HB1 = (gtsam::Matrix11() << -(YY*B*B + YY*AN*AT + 2*XX*B*AT) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();

            HXX2 = (gtsam::Matrix11() << B / (pow(B,2) - AN*AT) / pow(F_max,2)).finished();
            HYY2 = (gtsam::Matrix11() << AN / (pow(B,2) - AN*AT) / pow(F_max,2)).finished();
            HAT2 = (gtsam::Matrix11() << (YY*AN*AN + XX*B*AN) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();
            HAN2 = (gtsam::Matrix11() << (YY*B*B + XX*B*AT) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();
            HB2 = (gtsam::Matrix11() << -(XX*B*B + XX*AN*AT + 2*YY*B*AN) / pow((pow(B,2) - AN*AT),2) / pow(F_max,2)).finished();

        } else if (mu < -mu_max) {
            x_slip = mu_max * YY / (AT + mu_max*B) / pow(F_max,2);
            y_slip = - YY / (AT + mu_max*B) / pow(F_max,2);
            
            HXX1 = (gtsam::Matrix11() << 0).finished();
            HYY1 = (gtsam::Matrix11() << mu_max / (AT + mu_max*B) / pow(F_max,2)).finished();
            HAT1 = (gtsam::Matrix11() << - mu_max * YY / pow(AT + mu_max*B,2) / pow(F_max,2)).finished();
            HAN1 = (gtsam::Matrix11() << 0).finished();
            HB1 = (gtsam::Matrix11() << - mu_max * mu_max * YY / pow(AT + mu_max*B,2) / pow(F_max,2)).finished();

            HXX2 = (gtsam::Matrix11() << 0).finished();
            HYY2 = (gtsam::Matrix11() << - 1 / (AT + mu_max*B) / pow(F_max,2)).finished();
            HAT2 = (gtsam::Matrix11() << YY / pow(AT + mu_max*B,2) / pow(F_max,2)).finished();
            HAN2 = (gtsam::Matrix11() << 0).finished();
            HB2 = (gtsam::Matrix11() << mu_max * YY / pow(AT + mu_max*B,2) / pow(F_max,2)).finished();

        } else {
            x_slip = mu_max * YY / (mu_max*B - AT) / pow(F_max,2);
            y_slip = YY / (mu_max*B - AT) / pow(F_max,2);
            
            HXX1 = (gtsam::Matrix11() << 0).finished();
            HYY1 = (gtsam::Matrix11() << mu_max / (mu_max*B - AT) / pow(F_max,2)).finished();
            HAT1 = (gtsam::Matrix11() << mu_max * YY / pow(mu_max*B - AT,2) / pow(F_max,2)).finished();
            HAN1 = (gtsam::Matrix11() << 0).finished();
            HB1 = (gtsam::Matrix11() << - mu_max * mu_max * YY / pow(mu_max*B - AT,2) / pow(F_max,2)).finished();

            HXX2 = (gtsam::Matrix11() << 0).finished();
            HYY2 = (gtsam::Matrix11() << 1 / (mu_max*B - AT) / pow(F_max,2)).finished();
            HAT2 = (gtsam::Matrix11() << YY / pow(mu_max*B - AT,2) / pow(F_max,2)).finished();
            HAN2 = (gtsam::Matrix11() << 0).finished();
            HB2 = (gtsam::Matrix11() << - mu_max * YY / pow(mu_max*B - AT,2) / pow(F_max,2)).finished();
        }
  
        double theta_slip = (x_slip*cg_trn[1] - y_slip*cg_trn[0]) * pow(F_max,2) / pow(M_max,2);
        gtsam::Matrix Hcgt2 = (gtsam::Matrix13() << -y_slip * pow(F_max,2) / pow(M_max,2), x_slip * pow(F_max,2) / pow(M_max,2), 0).finished();
        gtsam::Matrix Hxs1 = (gtsam::Matrix11() << cg_trn[1] * pow(F_max,2) / pow(M_max,2)).finished();
        gtsam::Matrix Hys1 = (gtsam::Matrix11() << -cg_trn[0] * pow(F_max,2) / pow(M_max,2)).finished();        

        // Output
        gtsam::Vector output = (gtsam::Vector3() << theta_slip+slip_rot[2], x_slip+slip_trn[0], y_slip+slip_trn[1]).finished();
        gtsam::Matrix Hths = (gtsam::Matrix31() << 1, 0, 0).finished();
        gtsam::Matrix Hxs2 = (gtsam::Matrix31() << 0, 1, 0).finished();
        gtsam::Matrix Hys2 = (gtsam::Matrix31() << 0, 0, 1).finished();
        gtsam::Matrix Hsr = (gtsam::Matrix33() << 0, 0, 1,
                                                  0, 0, 0,
                                                  0, 0, 0).finished();
        gtsam::Matrix Hst = (gtsam::Matrix33() << 0, 0, 0,
                                                  1, 0, 0,
                                                  0, 1, 0).finished();


        gtsam::Matrix H_theta_slip = Hths;
        gtsam::Matrix H_x_slip = Hxs2;
        gtsam::Matrix H_y_slip = Hys2;
        gtsam::Matrix H_slip_rot = Hsr;
        gtsam::Matrix H_slip_trn = Hst;
        gtsam::Matrix HXX = H_x_slip * HXX1 + H_y_slip * HXX2;
        gtsam::Matrix HYY = H_x_slip * HYY1 + H_y_slip * HYY2;
        gtsam::Matrix HAT = H_x_slip * HAT1 + H_y_slip * HAT2;
        gtsam::Matrix HAN = H_x_slip * HAN1 + H_y_slip * HAN2;
        gtsam::Matrix HB = H_x_slip * HB1 + H_y_slip * HB2;
        gtsam::Matrix H_slip_g = H_slip_rot * Hsrg * Hrot + H_slip_trn * Hstg * Htrn;
        gtsam::Matrix HgG_rot = HXX * Hgr1 + HYY * Hgr2;
        gtsam::Matrix HgG_trn = HXX * Hgt1 + HYY * Hgt2;
        gtsam::Matrix HgG_g = HgG_rot * Hgrg * Hrot + HgG_trn * Hgtg * Htrn;
        gtsam::Matrix Hpcgr = H_slip_rot * Hcgr1 + HgG_rot * Hcgr2 + H_slip_trn * Hcgr3 + HgG_trn * Hcgr4;
        gtsam::Matrix Hcg_trn = H_theta_slip*Hcgt2 + HXX*Hcgt3 + HYY*Hcgt4 + HAT*Hcgt5 + HAN*Hcgt6 + HB*Hcgt7;
        gtsam::Matrix Hpcg = Hpcgr*Hcg1 + Hcg_trn*Hcg2;
        gtsam::Matrix Hpog = H_slip_g*Hog;
        gtsam::Matrix HpOG = H_slip_g*HOG;
        
        if (Ho) *Ho = Hpog * Ho1;
        if (HO) *HO = HpOG * HO1;
        if (Hg) *Hg = Hpog*Hg1 + HgG_g*Hg2 + Hpcg*Hg3;
        if (HG) *HG = HpOG*HG1 + HgG_g*HG2;
        if (Hc) *Hc = Hpcg*Hc1;

        return output;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 5;
  }

}; // \class PhysicsEquation

} /// namespace gtsam_custom_factors