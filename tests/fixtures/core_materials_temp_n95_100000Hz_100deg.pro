Include "Parameter.pro";
Function{
  b = {0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 1} ;
  mu_real = {3000.0, 2998.0, 2997.0, 2995.0, 2978.0, 2973.0, 1.0, 1.0} ;
  mu_imag = {1.0, 115.0, 130.00000000000003, 180.0, 360.0, 400.0, 400.0, 400.0} ;
  mu_imag_couples = ListAlt[b(), mu_imag()] ;
  mu_real_couples = ListAlt[b(), mu_real()] ;
  f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{List[mu_imag_couples]};
  f_mu_real_d[] = InterpolationLinear[Norm[$1]]{List[mu_real_couples]};
  f_mu_imag[] = f_mu_imag_d[$1];
  f_mu_real[] = f_mu_real_d[$1];
 }  