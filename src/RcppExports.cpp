// Despite the warning below, I still edited this file to make some legacy piece of code work.
// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// RIT_1class
SEXP RIT_1class(SEXP z, NumericVector weights, int L, int branch, int depth, int n_trees, int min_inter_sz, int n_cores, bool is_sparse);
RcppExport SEXP iRF_RIT_1class(SEXP zSEXP, SEXP weightsSEXP, SEXP LSEXP, SEXP branchSEXP, SEXP depthSEXP, SEXP n_treesSEXP, SEXP min_inter_szSEXP, SEXP n_coresSEXP, SEXP is_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type z(zSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< int >::type L(LSEXP);
    Rcpp::traits::input_parameter< int >::type branch(branchSEXP);
    Rcpp::traits::input_parameter< int >::type depth(depthSEXP);
    Rcpp::traits::input_parameter< int >::type n_trees(n_treesSEXP);
    Rcpp::traits::input_parameter< int >::type min_inter_sz(min_inter_szSEXP);
    Rcpp::traits::input_parameter< int >::type n_cores(n_coresSEXP);
    Rcpp::traits::input_parameter< bool >::type is_sparse(is_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(RIT_1class(z, weights, L, branch, depth, n_trees, min_inter_sz, n_cores, is_sparse));
    return rcpp_result_gen;
END_RCPP
}
// RIT_2class
SEXP RIT_2class(SEXP z, SEXP z0, int L, int branch, int depth, int n_trees, double theta0, double theta1, int min_inter_sz, int n_cores, bool is_sparse);
RcppExport SEXP iRF_RIT_2class(SEXP zSEXP, SEXP z0SEXP, SEXP LSEXP, SEXP branchSEXP, SEXP depthSEXP, SEXP n_treesSEXP, SEXP theta0SEXP, SEXP theta1SEXP, SEXP min_inter_szSEXP, SEXP n_coresSEXP, SEXP is_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type z(zSEXP);
    Rcpp::traits::input_parameter< SEXP >::type z0(z0SEXP);
    Rcpp::traits::input_parameter< int >::type L(LSEXP);
    Rcpp::traits::input_parameter< int >::type branch(branchSEXP);
    Rcpp::traits::input_parameter< int >::type depth(depthSEXP);
    Rcpp::traits::input_parameter< int >::type n_trees(n_treesSEXP);
    Rcpp::traits::input_parameter< double >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< double >::type theta1(theta1SEXP);
    Rcpp::traits::input_parameter< int >::type min_inter_sz(min_inter_szSEXP);
    Rcpp::traits::input_parameter< int >::type n_cores(n_coresSEXP);
    Rcpp::traits::input_parameter< bool >::type is_sparse(is_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(RIT_2class(z, z0, L, branch, depth, n_trees, theta0, theta1, min_inter_sz, n_cores, is_sparse));
    return rcpp_result_gen;
END_RCPP
}
// ancestorPath
NumericVector ancestorPath(DataFrame treeInfo, IntegerVector nodeVarIndices, int p, int nleaf, bool firstSplit);
RcppExport SEXP iRF_ancestorPath(SEXP treeInfoSEXP, SEXP nodeVarIndicesSEXP, SEXP pSEXP, SEXP nleafSEXP, SEXP firstSplitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< DataFrame >::type treeInfo(treeInfoSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nodeVarIndices(nodeVarIndicesSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type nleaf(nleafSEXP);
    Rcpp::traits::input_parameter< bool >::type firstSplit(firstSplitSEXP);
    rcpp_result_gen = Rcpp::wrap(ancestorPath(treeInfo, nodeVarIndices, p, nleaf, firstSplit));
    return rcpp_result_gen;
END_RCPP
}
