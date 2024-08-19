/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/libmints/multipolepotential.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/potential.h"
#include "typedefs.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/psifiles.h"
#include "psi4/libpsi4util/libpsi4util.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"

#include <libint2/shell.h>
#include <libint2/boys.h>
#include <libint2/engine.h>

using namespace psi;
using namespace mdintegrals;

MultipolePotentialInt::MultipolePotentialInt(std::vector<SphericalTransform>& spherical_transforms,
                                             std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2, int order,
                                             int deriv)
    : OneBodyAOInt(spherical_transforms, bs1, bs2, deriv), MDHelper(bs1->max_am(), bs2->max_am()), order_(order) {
    int maxnao1 = INT_NCART(maxam1_);
    int maxnao2 = INT_NCART(maxam2_);

    if (deriv > 0) {
        throw FeatureNotImplemented("LibMints", "MultipolePotentialInts called with deriv > 0", __FILE__, __LINE__);
    }

    // pre-allocate R matrix
    int am = maxam1_ + maxam2_;
    int rdim1 = am + order_ + 1;
    int rdim2 = rdim1 * rdim1 * rdim1;
    R = std::vector<double>(rdim1 * rdim2);

    // set up Boys function evaluator
    fm_eval_ = libint2::FmEval_Chebyshev7<double>::instance(am + order_);

    comps_der_ = std::vector<std::vector<std::array<int, 3>>>(order_ + 1);
    for (int d = 0; d < order_ + 1; ++d) {
        comps_der_[d] = generate_am_components_cca(d);
    }
    int nchunks = cumulative_cart_dim(order_);
    buffer_ = new double[nchunks * maxnao1 * maxnao2];
    set_chunks(nchunks);
    buffers_.resize(nchunk_);
}

MultipolePotentialInt::~MultipolePotentialInt() { delete[] buffer_; }

void MultipolePotentialInt::compute_pair(const libint2::Shell& s1, const libint2::Shell& s2) {

    outfile->Printf(" test compute pair \n");

    double Cx = origin_[0];
    double Cy = origin_[1];
    double Cz = origin_[2];
    Point C{Cx, Cy, Cz};

    int am1 = s1.contr[0].l;
    int am2 = s2.contr[0].l;
    int am = am1 + am2;

    const auto& comps_am1 = am_comps_[am1];
    const auto& comps_am2 = am_comps_[am2];

    auto A = s1.O;
    auto B = s2.O;
    int nprim1 = s1.nprim();
    int nprim2 = s2.nprim();

    // output buffer dimensions
    int dim1 = INT_NCART(am1);
    int dim2 = INT_NCART(am2);
    int size = dim1 * dim2;
    memset(buffer_, 0, nchunk_ * size * sizeof(double));

    // R matrix dimensions
    int r_am = am + order_;
    int rdim1 = r_am + 1;

    // E matrix dimensions
    int edim2 = am2 + 1;
    int edim3 = am1 + am2 + 2;

    outfile->Printf(" test1 compute pair \n");

    int ao12 = 0;
    for (int p1 = 0; p1 < nprim1; ++p1) {
        double a = s1.alpha[p1];
        double ca = s1.contr[0].coeff[p1];
        for (int p2 = 0; p2 < nprim2; ++p2) {
            double b = s2.alpha[p2];
            double cb = s2.contr[0].coeff[p2];

            double p = a + b;
            Point P{(a * A[0] + b * B[0]) / p, (a * A[1] + b * B[1]) / p, (a * A[2] + b * B[2]) / p};
            double prefac = 2.0 * M_PI * ca * cb / p;

            fill_E_matrix(am1, am2, P, A, B, a, b, Ex, Ey, Ez);

            outfile->Printf(" test2 compute pair \n");

            fill_R_matrix(r_am, p, P, C, R, fm_eval_);

            outfile->Printf(" test3 compute pair \n");

            int der_count = 0;
            double sign_prefac = prefac;
            // loop over 1/R derivatives
            for (int der = 0; der < order_ + 1; ++der) {
                const auto& comps = comps_der_[der];
                // loop over Cartesian components of the derivative
                // TODO: use structured bindings again once C++17 issues
                // with l2 are sorted out
                for (const auto& comp_der : comps) {
                    const auto ex = comp_der[0];
                    const auto ey = comp_der[1];
                    const auto ez = comp_der[2];
                    ao12 = 0;
                    for (const auto& comp_am1 : comps_am1) {
                        const auto l1 = comp_am1[0];
                        const auto m1 = comp_am1[1];
                        const auto n1 = comp_am1[2];
                        for (const auto& comp_am2 : comps_am2) {
                            const auto l2 = comp_am2[0];
                            const auto m2 = comp_am2[1];
                            const auto n2 = comp_am2[2];
                            double val = 0.0;
                            int maxt = l1 + l2;
                            int maxu = m1 + m2;
                            int maxv = n1 + n2;
                            // first two indices are already known, so avoid
                            // re-computing the entire address_3d
                            const double* Ex_p = &Ex.data()[edim3 * (l2 + edim2 * l1)];
                            const double* Ey_p = &Ey.data()[edim3 * (m2 + edim2 * m1)];
                            const double* Ez_p = &Ez.data()[edim3 * (n2 + edim2 * n1)];
                            for (int t = 0; t <= maxt; ++t) {
                                for (int u = 0; u <= maxu; ++u) {
                                    for (int v = 0; v <= maxv; ++v) {
                                        // eq 9.9.32 (using eq 9.9.27)
                                        val += Ex_p[t] * Ey_p[u] * Ez_p[v] *
                                               R[address_3d(t + ex, u + ey, v + ez, rdim1, rdim1)];
                                    }
                                }
                            }
                            buffer_[ao12 + size * der_count] += sign_prefac * val;
                            ++ao12;
                        }
                    }
                    der_count++;
                }
                // sign = (-1)^der
                sign_prefac *= -1.0;
            }
        }
    }
    pure_transform(s1, s2, cumulative_cart_dim(order_));
    for (int chunk = 0; chunk < cumulative_cart_dim(order_); ++chunk) {
        buffers_[chunk] = buffer_ + chunk * s1.size() * s2.size();
    }
}

MultipolePotentialInt_reg::MultipolePotentialInt_reg(double eta, std::vector<SphericalTransform>& spherical_transforms,
                                             std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2, int order,
                                             int deriv)
    : OneBodyAOInt(spherical_transforms, bs1, bs2, deriv), MDHelper(bs1->max_am(), bs2->max_am()), order_(order) {
    eta_ = eta;
    int maxnao1 = INT_NCART(maxam1_);
    int maxnao2 = INT_NCART(maxam2_);
    
 //   outfile->Printf(" test0 initialized reg \n");

    if (deriv > 0) {
        throw FeatureNotImplemented("LibMints", "MultipolePotentialInts called with deriv > 0", __FILE__, __LINE__);
    }

    // pre-allocate R matrix
    int am = maxam1_ + maxam2_;
    int rdim1 = am + order_ + 1;
    int rdim2 = rdim1 * rdim1 * rdim1;
    R = std::vector<double>(rdim1 * rdim2);


    // set up Boys function evaluator
    fm_eval_ = libint2::FmEval_Chebyshev7<double>::instance(am + order_);

    comps_der_ = std::vector<std::vector<std::array<int, 3>>>(order_ + 1);
    for (int d = 0; d < order_ + 1; ++d) {
        comps_der_[d] = generate_am_components_cca(d);
    }
    int nchunks = cumulative_cart_dim(order_);
    buffer_ = new double[nchunks * maxnao1 * maxnao2];
    set_chunks(nchunks);
    buffers_.resize(nchunk_);

    int max_am = std::max(bs1->max_am(), bs2->max_am());
    int max_nprim = std::max(bs1->max_nprimitive(), bs2->max_nprimitive());

    std::vector<std::pair<double, std::array<double, 3>>> Zxyz;
    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
        Zxyz.push_back({
            (double)bs1_->molecule()->Z(A),
            {bs1_->molecule()->x(A), bs1_->molecule()->y(A), bs1_->molecule()->z(A)}});
    }
    Zxyz_ = Zxyz;

    //engine0_ = std::make_unique<libint2::Engine>(libint2::Operator::nuclear, max_nprim, max_am, 0);
    //engine0_->set_params(Zxyz);

    // Setup the initial field of partial charges
//    C = std::make_shared<Matrix>("Partial Charge Field (Z,x,y,z)", bs1_->molecule()->natom(), 4);
//    double **Cp = C->pointer();

//    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
//        Cp[A][0] = (double)bs1_->molecule()->Z(A);
//        Cp[A][1] = bs1_->molecule()->x(A);
//        Cp[A][2] = bs1_->molecule()->y(A);
//        Cp[A][3] = bs1_->molecule()->z(A);
//    }

}

MultipolePotentialInt_reg::~MultipolePotentialInt_reg() { delete[] buffer_; }

void MultipolePotentialInt_reg::set_charge_field(std::vector<std::pair<double, std::array<double, 3>>>& Zxyz) {
//  engine0_->set_params(Zxyz);
    Zxyz_ = Zxyz;
}

void MultipolePotentialInt_reg::compute_pair_reg(double eta, const libint2::Shell& s1, const libint2::Shell& s2) {

//    outfile->Printf(" test3 compute pair reg \n");
 
 //   double Cx = origin_[0];
 //   double Cy = origin_[1];
 //   double Cz = origin_[2];
 //   Point C{Cx, Cy, Cz};

    int am1 = s1.contr[0].l;
    int am2 = s2.contr[0].l;
    int am = am1 + am2;

    const auto& comps_am1 = am_comps_[am1];
    const auto& comps_am2 = am_comps_[am2];

    auto A = s1.O;
    auto B = s2.O;
    int nprim1 = s1.nprim();
    int nprim2 = s2.nprim();

    // output buffer dimensions
    int dim1 = INT_NCART(am1);
    int dim2 = INT_NCART(am2);
    int size = dim1 * dim2;
    memset(buffer_, 0, nchunk_ * size * sizeof(double));

    // R matrix dimensions
    int r_am = am + order_;
    int rdim1 = r_am + 1;

    // E matrix dimensions
    int edim2 = am2 + 1;
    int edim3 = am1 + am2 + 2;

    int ao12 = 0;

//    double **Cp = C->pointer();
    int ncharge = Zxyz_.size();

    for (int p1 = 0; p1 < nprim1; ++p1) {
        double a = s1.alpha[p1];
        double ca = s1.contr[0].coeff[p1];
        for (int p2 = 0; p2 < nprim2; ++p2) {
            double b = s2.alpha[p2];
            double cb = s2.contr[0].coeff[p2];

            double p = a + b;
            Point P{(a * A[0] + b * B[0]) / p, (a * A[1] + b * B[1]) / p, (a * A[2] + b * B[2]) / p};
            double prefac = 2.0 * M_PI * ca * cb / p;

            fill_E_matrix(am1, am2, P, A, B, a, b, Ex, Ey, Ez);
 //            fill_R_matrix(r_am, p, P, C, R, fm_eval_);


            for (int atom = 0; atom < ncharge; ++atom) {
                double P_C[3];

                double Z = Zxyz_[atom].first;

                P_C[0] = P[0] - Zxyz_[atom].second[0];
                P_C[1] = P[1] - Zxyz_[atom].second[1];
                P_C[2] = P[2] - Zxyz_[atom].second[2];

                Point PC{P_C[0],P_C[1],P_C[2]};

                fill_R_matrix_reg(r_am, p, eta, PC, R, fm_eval_);
    
    
                int der_count = 0;
                double sign_prefac = prefac;
                // loop over 1/R derivatives
                for (int der = 0; der < order_ + 1; ++der) {
                    const auto& comps = comps_der_[der];
                    // loop over Cartesian components of the derivative
                    // TODO: use structured bindings again once C++17 issues
                    // with l2 are sorted out
                    for (const auto& comp_der : comps) {
                        const auto ex = comp_der[0];
                        const auto ey = comp_der[1];
                        const auto ez = comp_der[2];
                        ao12 = 0;
                        for (const auto& comp_am1 : comps_am1) {
                            const auto l1 = comp_am1[0];
                            const auto m1 = comp_am1[1];
                            const auto n1 = comp_am1[2];
                            for (const auto& comp_am2 : comps_am2) {
                                const auto l2 = comp_am2[0];
                                const auto m2 = comp_am2[1];
                                const auto n2 = comp_am2[2];
                                double val = 0.0;
                                int maxt = l1 + l2;
                                int maxu = m1 + m2;
                                int maxv = n1 + n2;
                                // first two indices are already known, so avoid
                                // re-computing the entire address_3d
                                const double* Ex_p = &Ex.data()[edim3 * (l2 + edim2 * l1)];
                                const double* Ey_p = &Ey.data()[edim3 * (m2 + edim2 * m1)];
                                const double* Ez_p = &Ez.data()[edim3 * (n2 + edim2 * n1)];
                                for (int t = 0; t <= maxt; ++t) {
                                    for (int u = 0; u <= maxu; ++u) {
                                        for (int v = 0; v <= maxv; ++v) {
                                            // eq 9.9.32 (using eq 9.9.27)
                                            val += Ex_p[t] * Ey_p[u] * Ez_p[v] *
                                                   R[address_3d(t + ex, u + ey, v + ez, rdim1, rdim1)];
                                        }
                                    }
                                }
                                buffer_[ao12 + size * der_count] += sign_prefac * val * -Z;
                                ++ao12;
                            }
                        }
                        der_count++;
                    }
                    // sign = (-1)^der
                    sign_prefac *= -1.0;
                }
            }
        }
    }


    pure_transform(s1, s2, cumulative_cart_dim(order_));
    for (int chunk = 0; chunk < cumulative_cart_dim(order_); ++chunk) {
        buffers_[chunk] = buffer_ + chunk * s1.size() * s2.size();
    }

}

MultipolePotentialInt_erf::MultipolePotentialInt_erf(double omega, std::vector<SphericalTransform>& spherical_transforms,
                                             std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2, int order,
                                             int deriv)
    : OneBodyAOInt(spherical_transforms, bs1, bs2, deriv), MDHelper(bs1->max_am(), bs2->max_am()), order_(order) {
    omega_ = omega;
    int maxnao1 = INT_NCART(maxam1_);
    int maxnao2 = INT_NCART(maxam2_);
    
    //outfile->Printf(" test0 initialized erf \n");

    if (deriv > 0) {
        throw FeatureNotImplemented("LibMints", "MultipolePotentialInts called with deriv > 0", __FILE__, __LINE__);
    }

    // pre-allocate R matrix
    int am = maxam1_ + maxam2_;
    int rdim1 = am + order_ + 1;
    int rdim2 = rdim1 * rdim1 * rdim1;
    R = std::vector<double>(rdim1 * rdim2);


    // set up Boys function evaluator
    fm_eval_ = libint2::FmEval_Chebyshev7<double>::instance(am + order_);

    comps_der_ = std::vector<std::vector<std::array<int, 3>>>(order_ + 1);
    for (int d = 0; d < order_ + 1; ++d) {
        comps_der_[d] = generate_am_components_cca(d);
    }
    int nchunks = cumulative_cart_dim(order_);
    buffer_ = new double[nchunks * maxnao1 * maxnao2];
    set_chunks(nchunks);
    buffers_.resize(nchunk_);

    int max_am = std::max(bs1->max_am(), bs2->max_am());
    int max_nprim = std::max(bs1->max_nprimitive(), bs2->max_nprimitive());

    std::vector<std::pair<double, std::array<double, 3>>> Zxyz;
    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
        Zxyz.push_back({
            (double)bs1_->molecule()->Z(A),
            {bs1_->molecule()->x(A), bs1_->molecule()->y(A), bs1_->molecule()->z(A)}});
    }
    Zxyz_ = Zxyz;

    //engine0_ = std::make_unique<libint2::Engine>(libint2::Operator::nuclear, max_nprim, max_am, 0);
    //engine0_->set_params(Zxyz);

    // Setup the initial field of partial charges
//    C = std::make_shared<Matrix>("Partial Charge Field (Z,x,y,z)", bs1_->molecule()->natom(), 4);
//    double **Cp = C->pointer();
//
//    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
//        Cp[A][0] = (double)bs1_->molecule()->Z(A);
//        Cp[A][1] = bs1_->molecule()->x(A);
//        Cp[A][2] = bs1_->molecule()->y(A);
//        Cp[A][3] = bs1_->molecule()->z(A);
//    }

}

MultipolePotentialInt_erf::~MultipolePotentialInt_erf() { delete[] buffer_; }

void MultipolePotentialInt_erf::set_charge_field(std::vector<std::pair<double, std::array<double, 3>>>& Zxyz) {
//  engine0_->set_params(Zxyz);
    Zxyz_ = Zxyz;
}

void MultipolePotentialInt_erf::compute_pair_erf(double omega, const libint2::Shell& s1, const libint2::Shell& s2) {

    //outfile->Printf(" In specific MultipolePotentialInt_erf::compute_pair_erf %d %d \n",s1,s2);
 
  //   double Cx = origin_[0];
 //   double Cy = origin_[1];
 //   double Cz = origin_[2];
 //   Point C{Cx, Cy, Cz};

    int am1 = s1.contr[0].l;
    int am2 = s2.contr[0].l;
    int am = am1 + am2;

    const auto& comps_am1 = am_comps_[am1];
    const auto& comps_am2 = am_comps_[am2];

    auto A = s1.O;
    auto B = s2.O;
    int nprim1 = s1.nprim();
    int nprim2 = s2.nprim();

    // output buffer dimensions
    int dim1 = INT_NCART(am1);
    int dim2 = INT_NCART(am2);
    int size = dim1 * dim2;
    memset(buffer_, 0, nchunk_ * size * sizeof(double));

    // R matrix dimensions
    int r_am = am + order_;
    int rdim1 = r_am + 1;

    // E matrix dimensions
    int edim2 = am2 + 1;
    int edim3 = am1 + am2 + 2;

    int ao12 = 0;

//    double **Cp = C->pointer();
//    int ncharge = C->rowspi()[0];

    int ncharge = Zxyz_.size();

    for (int p1 = 0; p1 < nprim1; ++p1) {
        double a = s1.alpha[p1];
        double ca = s1.contr[0].coeff[p1];
        for (int p2 = 0; p2 < nprim2; ++p2) {
            double b = s2.alpha[p2];
            double cb = s2.contr[0].coeff[p2];

            double p = a + b;
            Point P{(a * A[0] + b * B[0]) / p, (a * A[1] + b * B[1]) / p, (a * A[2] + b * B[2]) / p};
            double prefac = 2.0 * M_PI * ca * cb / p;

            fill_E_matrix(am1, am2, P, A, B, a, b, Ex, Ey, Ez);


            for (int atom = 0; atom < ncharge; ++atom) {
                double P_C[3];

                double Z = Zxyz_[atom].first;

                P_C[0] = P[0] - Zxyz_[atom].second[0];
                P_C[1] = P[1] - Zxyz_[atom].second[1];
                P_C[2] = P[2] - Zxyz_[atom].second[2];

                outfile->Printf("Z, P_C[0], P_C[1], P_C[2] = %12.6f %12.6f %12.6f %12.6f \n",Z,P_C[0], P_C[1], P_C[2]);
                Point PC{P_C[0],P_C[1],P_C[2]};

                fill_R_matrix_erf(r_am, p, omega, PC, R, fm_eval_);
    
    
                int der_count = 0;
                double sign_prefac = prefac;
                // loop over 1/R derivatives
                for (int der = 0; der < order_ + 1; ++der) {
                    const auto& comps = comps_der_[der];
                    // loop over Cartesian components of the derivative
                    // TODO: use structured bindings again once C++17 issues
                    // with l2 are sorted out
                    for (const auto& comp_der : comps) {
                        const auto ex = comp_der[0];
                        const auto ey = comp_der[1];
                        const auto ez = comp_der[2];
                        ao12 = 0;
                        for (const auto& comp_am1 : comps_am1) {
                            const auto l1 = comp_am1[0];
                            const auto m1 = comp_am1[1];
                            const auto n1 = comp_am1[2];
                            for (const auto& comp_am2 : comps_am2) {
                                const auto l2 = comp_am2[0];
                                const auto m2 = comp_am2[1];
                                const auto n2 = comp_am2[2];
                                double val = 0.0;
                                int maxt = l1 + l2;
                                int maxu = m1 + m2;
                                int maxv = n1 + n2;
                                // first two indices are already known, so avoid
                                // re-computing the entire address_3d
                                const double* Ex_p = &Ex.data()[edim3 * (l2 + edim2 * l1)];
                                const double* Ey_p = &Ey.data()[edim3 * (m2 + edim2 * m1)];
                                const double* Ez_p = &Ez.data()[edim3 * (n2 + edim2 * n1)];
                                for (int t = 0; t <= maxt; ++t) {
                                    for (int u = 0; u <= maxu; ++u) {
                                        for (int v = 0; v <= maxv; ++v) {
                                            // eq 9.9.32 (using eq 9.9.27)
                                            val += Ex_p[t] * Ey_p[u] * Ez_p[v] *
                                                   R[address_3d(t + ex, u + ey, v + ez, rdim1, rdim1)];
                                        }
                                    }
                                }
                                buffer_[ao12 + size * der_count] += sign_prefac * val * -Z;
                                ++ao12;
                             //   if (!(Z == 0)) {
                             //       outfile->Printf("buffer_[ao12 + size * der_count]   %12.6f\n", buffer_[ao12 + size * der_count]);
                             //   }
                            }
                        }
                        der_count++;
                    }
                    // sign = (-1)^der
                    sign_prefac *= -1.0;
                }
            }
        }
    }


    pure_transform(s1, s2, cumulative_cart_dim(order_));
    for (int chunk = 0; chunk < cumulative_cart_dim(order_); ++chunk) {
        buffers_[chunk] = buffer_ + chunk * s1.size() * s2.size();
    }

}


MultipolePotentialInt_erfgau::MultipolePotentialInt_erfgau(double omega, std::vector<SphericalTransform>& spherical_transforms,
                                             std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2, int order,
                                             int deriv)
    : OneBodyAOInt(spherical_transforms, bs1, bs2, deriv), MDHelper(bs1->max_am(), bs2->max_am()), order_(order) {
    omega_ = omega;
    int maxnao1 = INT_NCART(maxam1_);
    int maxnao2 = INT_NCART(maxam2_);
    
    //outfile->Printf(" test0 initialized erf \n");

    if (deriv > 0) {
        throw FeatureNotImplemented("LibMints", "MultipolePotentialInts called with deriv > 0", __FILE__, __LINE__);
    }

    // pre-allocate R matrix
    int am = maxam1_ + maxam2_;
    int rdim1 = am + order_ + 1;
    int rdim2 = rdim1 * rdim1 * rdim1;
    R = std::vector<double>(rdim1 * rdim2);


    comps_der_ = std::vector<std::vector<std::array<int, 3>>>(order_ + 1);
    for (int d = 0; d < order_ + 1; ++d) {
        comps_der_[d] = generate_am_components_cca(d);
    }
    int nchunks = cumulative_cart_dim(order_);
    buffer_ = new double[nchunks * maxnao1 * maxnao2];
    set_chunks(nchunks);
    buffers_.resize(nchunk_);

    int max_am = std::max(bs1->max_am(), bs2->max_am());
    int max_nprim = std::max(bs1->max_nprimitive(), bs2->max_nprimitive());

    std::vector<std::pair<double, std::array<double, 3>>> Zxyz;
    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
        Zxyz.push_back({
            (double)bs1_->molecule()->Z(A),
            {bs1_->molecule()->x(A), bs1_->molecule()->y(A), bs1_->molecule()->z(A)}});
    }
    Zxyz_ = Zxyz;

    //engine0_ = std::make_unique<libint2::Engine>(libint2::Operator::nuclear, max_nprim, max_am, 0);
    //engine0_->set_params(Zxyz);

    // Setup the initial field of partial charges
//    C = std::make_shared<Matrix>("Partial Charge Field (Z,x,y,z)", bs1_->molecule()->natom(), 4);
//    double **Cp = C->pointer();
//
//    for (int A = 0; A < bs1_->molecule()->natom(); A++) {
//        Cp[A][0] = (double)bs1_->molecule()->Z(A);
//        Cp[A][1] = bs1_->molecule()->x(A);
//        Cp[A][2] = bs1_->molecule()->y(A);
//        Cp[A][3] = bs1_->molecule()->z(A);
//    }

}

MultipolePotentialInt_erfgau::~MultipolePotentialInt_erfgau() { delete[] buffer_; }

void MultipolePotentialInt_erfgau::set_charge_field(std::vector<std::pair<double, std::array<double, 3>>>& Zxyz) {
//  engine0_->set_params(Zxyz);
    Zxyz_ = Zxyz;
}

void MultipolePotentialInt_erfgau::compute_pair_erfgau(double omega, const libint2::Shell& s1, const libint2::Shell& s2) {

    //outfile->Printf(" In specific MultipolePotentialInt_erf::compute_pair_erf %d %d \n",s1,s2);
 
  //   double Cx = origin_[0];
 //   double Cy = origin_[1];
 //   double Cz = origin_[2];
 //   Point C{Cx, Cy, Cz};

    int am1 = s1.contr[0].l;
    int am2 = s2.contr[0].l;
    int am = am1 + am2;

    const auto& comps_am1 = am_comps_[am1];
    const auto& comps_am2 = am_comps_[am2];

    auto A = s1.O;
    auto B = s2.O;
    int nprim1 = s1.nprim();
    int nprim2 = s2.nprim();

    // output buffer dimensions
    int dim1 = INT_NCART(am1);
    int dim2 = INT_NCART(am2);
    int size = dim1 * dim2;
    memset(buffer_, 0, nchunk_ * size * sizeof(double));

    // R matrix dimensions
    int r_am = am + order_;
    int rdim1 = r_am + 1;

    // E matrix dimensions
    int edim2 = am2 + 1;
    int edim3 = am1 + am2 + 2;

    int ao12 = 0;

//    double **Cp = C->pointer();
//    int ncharge = C->rowspi()[0];

    int ncharge = Zxyz_.size();

    for (int p1 = 0; p1 < nprim1; ++p1) {
        double a = s1.alpha[p1];
        double ca = s1.contr[0].coeff[p1];
        for (int p2 = 0; p2 < nprim2; ++p2) {
            double b = s2.alpha[p2];
            double cb = s2.contr[0].coeff[p2];

            double p = a + b;
            double q = a * b;
            Point P{(a * A[0] + b * B[0]) / p, (a * A[1] + b * B[1]) / p, (a * A[2] + b * B[2]) / p};
            double u = a * b / (a + b);
            Point AB{A[0] - B[0], A[1] - B[1], A[2] - B[2]};
            //double prefac = 2.0 * M_PI * ca * cb / p;
            double prefac = ca * cb;
            outfile->Printf("a b ca cb %12.6f %12.6f %12.6f %12.6f \n",a,b,ca,cb);

            fill_E_matrix(am1, am2, P, A, B, a, b, Ex, Ey, Ez);


            for (int atom = 0; atom < ncharge; ++atom) {
                double P_C[3];

                double Z = Zxyz_[atom].first;

                P_C[0] = P[0] - Zxyz_[atom].second[0];
                P_C[1] = P[1] - Zxyz_[atom].second[1];
                P_C[2] = P[2] - Zxyz_[atom].second[2];

          //    outfile->Printf("Z, P_C[0], P_C[1], P_C[2] = %12.6f %12.6f %12.6f %12.6f \n",Z,P_C[0], P_C[1], P_C[2]);
                Point PC{P_C[0],P_C[1],P_C[2]};

                outfile->Printf("omega multipolepot. : %12.8f",omega);
                fill_R_matrix_erfgau(r_am, p, omega, PC, R, u, AB, q);
                outfile->Printf(" R[0] %12.8f\n",R[0]);
    
    
                int der_count = 0;
                double sign_prefac = prefac;
                // loop over 1/R derivatives
                for (int der = 0; der < order_ + 1; ++der) {
                    const auto& comps = comps_der_[der];
                    // loop over Cartesian components of the derivative
                    // TODO: use structured bindings again once C++17 issues
                    // with l2 are sorted out
                    for (const auto& comp_der : comps) {
                        const auto ex = comp_der[0];
                        const auto ey = comp_der[1];
                        const auto ez = comp_der[2];
                        ao12 = 0;
                        for (const auto& comp_am1 : comps_am1) {
                            const auto l1 = comp_am1[0];
                            const auto m1 = comp_am1[1];
                            const auto n1 = comp_am1[2];
                            for (const auto& comp_am2 : comps_am2) {
                                const auto l2 = comp_am2[0];
                                const auto m2 = comp_am2[1];
                                const auto n2 = comp_am2[2];
                                double val = 0.0;
                                int maxt = l1 + l2;
                                int maxu = m1 + m2;
                                int maxv = n1 + n2;
                                // first two indices are already known, so avoid
                                // re-computing the entire address_3d
                                const double* Ex_p = &Ex.data()[edim3 * (l2 + edim2 * l1)];
                                const double* Ey_p = &Ey.data()[edim3 * (m2 + edim2 * m1)];
                                const double* Ez_p = &Ez.data()[edim3 * (n2 + edim2 * n1)];
                                for (int t = 0; t <= maxt; ++t) {
                                    for (int u = 0; u <= maxu; ++u) {
                                        for (int v = 0; v <= maxv; ++v) {
                                            // eq 9.9.32 (using eq 9.9.27)
                                            val += Ex_p[t] * Ey_p[u] * Ez_p[v] *
                                                   R[address_3d(t + ex, u + ey, v + ez, rdim1, rdim1)];
                                        }
                                    }
                                }
                            //  buffer_[ao12 + size * der_count] += sign_prefac * val * -Z;
                                buffer_[ao12 + size * der_count] += sign_prefac * val;
                                //outfile->Printf("buffer_[ao12 + size * der_count]   %12.6f %12.6f %12.6f\n", buffer_[ao12 + size * der_count], sign_prefac, val);
                                ++ao12;
                            }
                        }
                        der_count++;
                    }
                    // sign = (-1)^der
                    sign_prefac *= -1.0;
                }
            }
        }
    }


    pure_transform(s1, s2, cumulative_cart_dim(order_));
    for (int chunk = 0; chunk < cumulative_cart_dim(order_); ++chunk) {
        buffers_[chunk] = buffer_ + chunk * s1.size() * s2.size();
        outfile->Printf("Still in compute_pair %12.8f\n",buffers_[chunk][0]);
    }

}


