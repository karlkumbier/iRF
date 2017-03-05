// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// nodeObs
IntegerVector nodeObs(IntegerVector obsnodes, int n, int ntree, IntegerVector nrnodes, IntegerVector nodeobs);
RcppExport SEXP iRF_nodeObs(SEXP obsnodesSEXP, SEXP nSEXP, SEXP ntreeSEXP, SEXP nrnodesSEXP, SEXP nodeobsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type obsnodes(obsnodesSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nrnodes(nrnodesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nodeobs(nodeobsSEXP);
    rcpp_result_gen = Rcpp::wrap(nodeObs(obsnodes, n, ntree, nrnodes, nodeobs));
    return rcpp_result_gen;
END_RCPP
}
// nodeVars
IntegerVector nodeVars(IntegerVector varnodes, int ntree, int nrnodes, IntegerVector parents, IntegerVector idcskeep, IntegerVector nodect, IntegerVector nnodest, IntegerVector nodevars);
RcppExport SEXP iRF_nodeVars(SEXP varnodesSEXP, SEXP ntreeSEXP, SEXP nrnodesSEXP, SEXP parentsSEXP, SEXP idcskeepSEXP, SEXP nodectSEXP, SEXP nnodestSEXP, SEXP nodevarsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type varnodes(varnodesSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    Rcpp::traits::input_parameter< int >::type nrnodes(nrnodesSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type parents(parentsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type idcskeep(idcskeepSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nodect(nodectSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nnodest(nnodestSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nodevars(nodevarsSEXP);
    rcpp_result_gen = Rcpp::wrap(nodeVars(varnodes, ntree, nrnodes, parents, idcskeep, nodect, nnodest, nodevars));
    return rcpp_result_gen;
END_RCPP
}
