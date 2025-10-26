/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "search.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <utility>

#include <vector>
#include "bitboard.h"
#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "ucioption.h"

/*
  NOTE (Bayesian search deltas vs master):
  - Adds BayesConfig and per-thread tables for p*(d), z*(d), σ(d).
  - Introduces bayes_probcut_gate() with integer-only Q8 comparisons to screen
    ProbCut candidates before verification (Step 11).
  - Adds small ProbCut variant (Step 12) parameterized via BayesSmallProbCutCp and
    BayesTTLBMinDepth.
  - Adds Beta–Binomial posterior-based z* shave, and multiple-tests correction.
  - Adds QS pre-gate v2 to cheaply screen losing captures in quiescence.
  These blocks are guarded by the BayesEnabled UCI switch and keep PV-correctness
  by falling back to master behavior when disabled.
*/


namespace Stockfish {


// Forward declarations needed by Worker::search()/qsearch() before helper definitions (parity with master).
namespace {
using SearchedList = ValueList<Move, 32>;
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(Search::Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Search::Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos,
                      Search::Stack*  ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove);
}  // namespace

// Helper: pre-move captured piece (handles EN PASSANT correctly)
static inline Piece CapturedPiecePre(const Position& pos, Move m) {
    Piece p = pos.piece_on(m.to_sq());
    if (m.type_of() == EN_PASSANT)
        p = make_piece(~pos.side_to_move(), PAWN);
    return p;
}

// Helper: value of a (possibly NO_PIECE) captured piece
static inline int ValueOfCapturedPiece(Piece p) {
    if (p == NO_PIECE)
        return 0;
    return PieceValue[p];
}


namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap&            options,
                      const Search::LimitsType&    limits,
                      Stockfish::Position&         pos,
                      Stockfish::Search::RootMove& rootMove,
                      Value&                       v);

using namespace Search;


// Common clamp for Q8 * cp intermediates in gating thresholds
// Cover 30,000cp * 256 = 7,680,000 to avoid saturation bias
constexpr long long Q8_RAIL = 8000000LL;
// Max generations lag to trust TT recency
// removed unused TT_GEN_LAG [[maybe_unused]]

inline int mul_q8_cp_clamped(int zQ8, int sigmaCp) {
    long long prod = 1LL * std::max(0, zQ8) * std::max(1, sigmaCp);
    if (prod > Q8_RAIL)
        return Q8_RAIL;
    return int(prod);
}
// (anonymous namespace removed for linkage consistency)
// ---------------- Bayesian ProbCut helpers (tunable via UCI) ---------------
struct BayesConfig {
    bool enabled;
    bool tbGuard;        // Disable Bayes gates near Syzygy / TB regimes
    bool debugCounters;  // Placeholder switch for counters
    int  pstarPermille, cdfMode, tNu;
    // z* biases / rails (Q8)
    int  zTTBiasQ8, zMidBiasQ8, zEndBiasQ8, zPvBiasQ8, zTTUBiasQ8, zTTBestBiasQ8;
    int  zImprovingBiasQ8, zNotImprovingBiasQ8, zStarMinQ8, zTightBiasQ8, tightWindowCp;
    bool tightOnPVOnly;

    // μ & hist / SEE & class biases (centipawns)
    int capWeightPct, histScaleDiv, histCapCp, muBiasCp, muPromoBiasCp, muMajorCapBiasCp;
    int muSEEBoostCp, muTTAdjDiv, muTTAdjCapCp;

    // σ anchors & scales (centipawns)
    int sigmaD0, sigmaD3, sigmaD5, sigmaD8, sigmaD12, sigmaD20, sigmaScalePct;
    int sigmaMidPct, sigmaEndPct, ttBoostCp, sigmaFloorCp, sigmaCeilCp;
    int sigmaVolThreshCp, sigmaVolAddCp, ttNearCp, ttNearBoostCp;
    int smallProbCutCp;  // Step-12 margin (Cp)

    // Gate controls & guards
    int  gateMinDepth, seeMarginCp, minCapturedValueCp, maxGatedCaptures, maxVerifPerNode;
    bool gateCutOnly, noGateInCheck;
    int  noGateAtDepth, noGateRule50;

    // Depth-aware p*(d) schedule
    int pstarShallowPermille, pstarDeepPermille, pstarShallowDepth, pstarDeepDepth;

    // EVI lower bound
    int eviMinCp;

    // Beta–Binomial posterior + mixing
    bool bbEnabled;
    int  bbAlpha0, bbBeta0, bbGainQ8, bbMaxBiasQ8, bbMixPermille, bbMinTrials;

    // QS v2
    int qsEnabled, qsSigmaCp, qsZStarQ8, qsAlphaMarginCp, qsMaxGatedCaptures, qsMinCapturedValueCp;

    // Multiple-Tests Correction (MTC)
    int zMtcStepQ8, zMtcMaxQ8;

    // TT LB strong min-depth (absolute)
    int ttLBMinDepth;


    int qsTTLBMinDepth;        // Step-11 ProbCut tunables (defaults match master)
    int probCutBetaBaseCp;     // default 224
    int probCutBetaImproveCp;  // default 64
    int probCutDepthOffset;    // default 5
};

inline BayesConfig load_bayes_config(const OptionsMap& o) {
    // Non-inserting getter to avoid creating phantom UCI options (and races)
    auto get = [&](const char* k, int /*def*/ = 0) -> int { return int(o[k]); };

    BayesConfig B{};
    B.enabled       = get("BayesEnabled", 0);
    B.tbGuard       = get("BayesTBGuard", 1);
    B.debugCounters = get("BayesDebugCounters", 0);
    if (!B.enabled)
        return B;
    B.pstarPermille = get("BayesPStarPermille", 920);
    B.cdfMode       = std::clamp(get("BayesCDFMode", 0), 0, 2);
    B.tNu =
      std::clamp(get("BayesTNu", 8), 3, 64);  // Student‑t requires ν>2 to have finite variance
    // Clamp defensive ranges that matter for math stability
    B.zTTBiasQ8           = get("BayesZTTBiasQ8", 0);
    B.zTTUBiasQ8          = get("BayesZTTUBiasQ8", 0);
    B.zMidBiasQ8          = get("BayesZMidBiasQ8", 0);
    B.zEndBiasQ8          = get("BayesZEndBiasQ8", 0);
    B.zPvBiasQ8           = get("BayesZPvBiasQ8", 0);
    B.zTTBestBiasQ8       = get("BayesZTTBestBiasQ8", 0);
    B.zImprovingBiasQ8    = get("BayesZImprovingBiasQ8", 0);
    B.zNotImprovingBiasQ8 = get("BayesZNotImprovingBiasQ8", 0);
    B.zStarMinQ8          = get("BayesZStarMinQ8", 0);
    B.zTightBiasQ8        = get("BayesZTightBiasQ8", 0);
    B.tightWindowCp       = get("BayesTightWindowCp", 0);
    B.tightOnPVOnly       = get("BayesTightOnPVOnly", 0);

    B.capWeightPct     = get("BayesCapWeightPct", 50);
    B.histScaleDiv     = std::max(1, get("BayesHistScaleDiv", 64));
    B.histCapCp        = get("BayesHistCapCp", 0);
    B.muBiasCp         = get("BayesMuBiasCp", 0);
    B.muPromoBiasCp    = get("BayesMuPromoBiasCp", 0);
    B.muMajorCapBiasCp = get("BayesMuMajorCapBiasCp", 0);
    B.muSEEBoostCp     = get("BayesMuSEEBoostCp", 0);
    B.muTTAdjDiv       = get("BayesMuTTAdjDiv", 0);
    B.muTTAdjCapCp     = get("BayesMuTTAdjCapCp", 0);

    B.sigmaD0          = std::max(1, get("BayesSigmaD0Cp", 200));
    B.sigmaD3          = get("BayesSigmaD3Cp", 220);
    B.sigmaD5          = get("BayesSigmaD5Cp", 240);
    B.sigmaD8          = get("BayesSigmaD8Cp", 260);
    B.sigmaD12         = get("BayesSigmaD12Cp", 280);
    B.sigmaD20         = get("BayesSigmaD20Cp", 320);
    B.sigmaScalePct    = get("BayesSigmaScalePct", 100);
    B.sigmaMidPct      = get("BayesSigmaMidPct", 100);
    B.sigmaEndPct      = get("BayesSigmaEndPct", 100);
    B.ttBoostCp        = get("BayesTTBoostCp", 0);
    B.sigmaFloorCp     = get("BayesSigmaFloorCp", 0);
    B.sigmaCeilCp      = get("BayesSigmaCeilCp", 0);
    B.sigmaVolThreshCp = get("BayesSigmaVolThreshCp", 0);
    B.sigmaVolAddCp    = get("BayesSigmaVolAddCp", 0);
    B.ttNearCp         = get("BayesTTNearCp", 64);
    B.ttNearBoostCp    = get("BayesTTNearBoostCp", 16);
    B.smallProbCutCp   = get("BayesSmallProbCutCp", 418);

    B.gateMinDepth       = get("BayesGateMinDepth", 4);
    B.seeMarginCp        = get("BayesSEEMarginCp", 0);
    B.minCapturedValueCp = get("BayesMinCapturedValueCp", 0);
    B.maxGatedCaptures   = get("BayesMaxGatedCaptures", 0);
    B.maxVerifPerNode    = get("BayesMaxVerifPerNode", 0);
    B.gateCutOnly        = get("BayesGateCutOnly", 0);
    B.noGateInCheck      = get("BayesNoGateInCheck", 0);
    B.noGateAtDepth      = get("BayesNoGateAtDepth", 0);
    B.noGateRule50       = get("BayesNoGateRule50", 0);

    B.pstarShallowPermille = get("BayesPStarShallowPermille", 0);
    B.pstarDeepPermille    = get("BayesPStarDeepPermille", 0);
    B.pstarShallowDepth    = get("BayesPStarShallowDepth", 4);
    B.pstarDeepDepth       = get("BayesPStarDeepDepth", 12);
    B.eviMinCp             = get("BayesEviMinCp", 0);

    B.bbEnabled     = get("BayesBBEnabled", 0);
    B.bbAlpha0      = get("BayesBBAlpha0", 1);
    B.bbBeta0       = get("BayesBBBeta0", 1);
    B.bbGainQ8      = get("BayesBBGainQ8", 0);
    B.bbMaxBiasQ8   = get("BayesBBMaxBiasQ8", 0);
    B.bbMixPermille = get("BayesBBMixPermille", 1000);
    B.bbMinTrials   = get("BayesBBMinTrials", 64);

    B.qsEnabled            = get("BayesQSEnabled", 0);
    B.qsSigmaCp            = get("BayesQSSigmaCp", 0);
    B.qsZStarQ8            = get("BayesQSZStarQ8", 360);
    B.qsAlphaMarginCp      = get("BayesQSAlphaMarginCp", 0);
    B.qsMaxGatedCaptures   = get("BayesQSMaxGatedCaptures", 2);
    B.qsMinCapturedValueCp = get("BayesQSMinCapturedValueCp", 0);

    B.zMtcStepQ8 = get("BayesZMtcStepQ8", 0);
    B.zMtcMaxQ8  = get("BayesZMtcMaxQ8", 0);

    B.ttLBMinDepth = get("BayesTTLBMinDepth", 0);


    B.qsTTLBMinDepth = get("BayesQSTTLBMinDepth", 0);
    // Step-11 ProbCut parameters (only used when BayesEnabled=true)
    B.probCutBetaBaseCp    = get("BayesProbCutBetaBaseCp", 224);
    B.probCutBetaImproveCp = get("BayesProbCutBetaImproveCp", 64);
    B.probCutDepthOffset   = get("BayesProbCutDepthOffset", 5);
    return B;
}

// Return true iff any Bayes feature will have observable effect vs master.
inline bool bayes_has_effect(const BayesConfig& B) {
    if (!B.enabled)
        return false;

    // Consider Bayes "effective" only when some setting can change pruning/order vs master.
    // A positive gateMinDepth alone must NOT flip Bayes on; otherwise enabling Bayes with
    // neutral knobs still adds overhead.
    const bool zstar_possible = (B.pstarPermille > 500) || (B.pstarShallowPermille > 500)
                             || (B.pstarDeepPermille > 500) || (B.zStarMinQ8 > 0)
                             || (B.zTTBiasQ8 > 0) || (B.zTTUBiasQ8 > 0) || (B.zPvBiasQ8 > 0)
                             || (B.zMidBiasQ8 > 0) || (B.zEndBiasQ8 > 0) || (B.zImprovingBiasQ8 > 0)
                             || (B.zNotImprovingBiasQ8 > 0) || (B.zTightBiasQ8 > 0);

    const bool gate_effective = (B.gateMinDepth > 0)  // the gate is allowed structurally
                             && zstar_possible;       // and could ever pass/fail something

    const bool qs_effective  = (B.qsEnabled != 0);
    const bool mtc_effective = (B.zMtcStepQ8 != 0);
    const bool bb_effective  = (B.bbEnabled != 0);
    const bool evi_effective = (B.eviMinCp > 0);

    if (gate_effective || qs_effective || mtc_effective || bb_effective || evi_effective)
        return true;

    // Deviations from master ProbCut constants are effects by themselves
    if (B.probCutBetaBaseCp != 224)
        return true;
    if (B.probCutBetaImproveCp != 64)
        return true;
    if (B.probCutDepthOffset != 5)
        return true;
    if (B.smallProbCutCp != 418)
        return true;

    // Otherwise, fully inert (no extra work, master‑equivalent).
    return false;
}


// p* -> z* in Q8. Cheap LUT + linear interpolation; optional logistic / Student-t adjustments.
inline int compute_zstar_q8(int pPermille, int mode, int tNu) {
    pPermille = std::min(999, std::max(500, pPermille));
    if (mode == 1)
    {  // Logistic (sd-matched)
        constexpr double PI = 3.14159265358979323846;
        const double     p  = pPermille / 1000.0;
        const double     s  = std::sqrt(3.0) / PI;
        const double     z  = s * std::log(p / (1.0 - p));
        return std::max(0, int(std::round(std::fabs(z) * 256.0)));
    }
    if (pPermille <= 800)
        return ((pPermille - 500) * 216) / 300;
    struct E {
        int p, zq8;
    };
    static constexpr E LUT[] = {{800, 216}, {850, 265}, {900, 328}, {920, 360},
                                {940, 397}, {950, 421}, {960, 448}, {970, 481},
                                {980, 525}, {990, 595}, {995, 660}, {999, 791}};
    int                i     = 0;
    constexpr int      LUT_N = int(sizeof(LUT) / sizeof(LUT[0]));
    while (i + 1 < LUT_N && LUT[i + 1].p <= pPermille)
        ++i;
    if (pPermille == LUT[i].p || i == LUT_N - 1)
        return LUT[i].zq8;
    const int p0 = LUT[i].p, p1 = LUT[i + 1].p, z0 = LUT[i].zq8, z1 = LUT[i + 1].zq8;
    int       zq8N = z0 + (z1 - z0) * (pPermille - p0) / std::max(1, p1 - p0);
    if (mode == 2)
    {  // Student‑t sd‑matched: scale > Normal
        const int    nu = std::max(3, tNu);
        const double c  = std::sqrt(double(nu) / double(nu - 2));
        zq8N            = int(std::round(zq8N * c));
    }
    return std::min(1024, std::max(0, zq8N));
}

inline int sigma_by_depth_linear(int d, const BayesConfig& c) {
    struct A {
        int depth, sigma;
    };
    // Hand-coded anchors to avoid a vector:
    const A a0{0, c.sigmaD0};
    const A a1{3, c.sigmaD3};
    const A a2{5, c.sigmaD5};
    const A a3{8, c.sigmaD8};
    const A a4{12, c.sigmaD12};
    const A a5{20, c.sigmaD20};
    auto    lerp = [](int x0, int y0, int x1, int y1, int x) {
        if (x <= x0)
            return y0;
        if (x >= x1)
            return y1;
        return y0 + (y1 - y0) * (x - x0) / std::max(1, x1 - x0);
    };
    int base = (d <= a1.depth) ? lerp(a0.depth, a0.sigma, a1.depth, a1.sigma, d)
             : (d <= a2.depth) ? lerp(a1.depth, a1.sigma, a2.depth, a2.sigma, d)
             : (d <= a3.depth) ? lerp(a2.depth, a2.sigma, a3.depth, a3.sigma, d)
             : (d <= a4.depth)
               ? lerp(a3.depth, a3.sigma, a4.depth, a4.sigma, d)
               : lerp(a4.depth, a4.sigma, a5.depth, a5.sigma, std::min(d, a5.depth));
    return std::max(1, base * c.sigmaScalePct / 100);
}

// Per-search tables (thread-local, small, L1-resident)
static thread_local int g_sigmaByDepth[32];
static thread_local int g_pstarByDepthPermille[32];
static thread_local int g_zstarByDepthQ8[32];
static thread_local int g_capWeightVal[PIECE_NB];
// Cached Bayes config for this thread/search iteration
static thread_local BayesConfig
  g_bayesCfg{};  // Zero-initialize to keep fields defined even when Bayes is disabled

// Beta–Binomial bucket (reduced-depth × phase) and global tried/success
static thread_local uint32_t g_bb_tried[6][3], g_bb_succ[6][3];
static thread_local uint32_t g_bb_tried_global = 0, g_bb_succ_global = 0;

// Saturating increment for 32-bit BB counters
inline void maybe_age_bb();
inline void sat_inc(uint32_t& x) {
    if (x < std::numeric_limits<uint32_t>::max())
        ++x;
}


// Gentle aging of Beta–Binomial counters to keep posterior responsive.
// Ages roughly every ~2^20 global trials by halving counts.
inline void maybe_age_bb() {
    if ((g_bb_tried_global & ((1u << 18) - 1)) == 0 && g_bb_tried_global)
    {
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 3; ++j)
            {
                g_bb_tried[i][j] >>= 1;
                g_bb_succ[i][j] >>= 1;
            }
        g_bb_tried_global >>= 1;
        g_bb_succ_global >>= 1;
    }
}
// Bucketing helpers
inline int bb_depth_bucket(int rd) {
    if (rd <= 2)
        return 0;
    if (rd == 3)
        return 1;
    if (rd <= 5)
        return 2;
    if (rd <= 8)
        return 3;
    if (rd <= 12)
        return 4;
    return 5;
}
inline int bb_phase_bucket(int npmBoth) {
    return (npmBoth <= 2400) ? 2 : (npmBoth <= 4800 ? 1 : 0);
}

// The gate itself:  integer‑only Q8 compare + optional EVI
inline bool bayes_probcut_gate(Value              staticEval,
                               Value              probCutBeta,
                               int                captHist,
                               int                capWeightPre,
                               int                muBiasNode,
                               int                thrQ8Node,
                               int                eviReqCp,
                               const BayesConfig& B) {
    int histTerm = captHist / B.histScaleDiv;
    if (B.histCapCp > 0)
    {
        if (histTerm > B.histCapCp)
            histTerm = B.histCapCp;
        if (histTerm < -B.histCapCp)
            histTerm = -B.histCapCp;
    }
    int muCp = int(staticEval) + capWeightPre + histTerm + B.muBiasCp + muBiasNode;
    muCp     = std::clamp(muCp, -30000, 30000);
    if (muCp <= int(probCutBeta))
        return false;
    if (eviReqCp >= 0)
    {
        const int marginCp = muCp - int(probCutBeta);
        if (marginCp < eviReqCp)
            return false;
    }
    // Compute in 64-bit then clamp, to be robust under aggressive compilation.
    const long long diffQ8ll =
      (static_cast<long long>(muCp) - static_cast<long long>(int(probCutBeta))) * 256LL;
    const int diffQ8 = int(std::clamp(diffQ8ll, -1LL * Q8_RAIL, 1LL * Q8_RAIL));
    // Compare to (z*·σ) in Q8·cp units
    return diffQ8 >= thrQ8Node;
}


// Bayes branch: keep the master container to avoid divergence.
// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

// (*Scaler) All tuned parameters at time controls shorter than
// optimized for require verifications at longer time controls

int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us     = pos.side_to_move();
    const auto  m      = (ss - 1)->currentMove;
    const auto  pcv    = w.pawnCorrectionHistory[pawn_correction_history_index(pos)][us];
    const auto  micv   = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto  wnpcv  = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto  bnpcv  = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const Piece pc_idx = m.is_ok()
                         ? (g_bayesCfg.enabled ? CapturedPiecePre(pos, m) : pos.piece_on(m.to_sq()))
                         : NO_PIECE;
    int        cnt2 = m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pc_idx][m.to_sq()] : 8;
    int        cnt4 = m.is_ok() ? (*(ss - 4)->continuationCorrectionHistory)[pc_idx][m.to_sq()] : 0;
    const auto cntcv = m.is_ok() ? (cnt2 + cnt4) : 8;

    return 9536 * pcv + 8494 * micv + 10132 * (wnpcv + bnpcv) + 7156 * cntcv;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos,
                               Stack* const    ss,
                               Search::Worker& workerThread,
                               const int       bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    constexpr int nonPawnWeight = 165;

    workerThread.pawnCorrectionHistory[pawn_correction_history_index(pos)][us] << bonus;
    workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 145 / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us]
      << bonus * nonPawnWeight / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us]
      << bonus * nonPawnWeight / 128;

    if (m.is_ok())
        (*(ss - 2)->continuationCorrectionHistory)[g_bayesCfg.enabled ? CapturedPiecePre(pos, m)
                                                                      : pos.piece_on(m.to_sq())]
                                                  [m.to_sq()]
          << bonus * 137 / 128;
}

// Add a small random component to draw evaluations to avoid 3-fold blindness
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
// namespace

Search::Worker::Worker(SharedState&                    sharedState,
                       std::unique_ptr<ISearchManager> sm,
                       size_t                          threadId,
                       NumaReplicatedAccessToken       token) :
    // Unpack the SharedState struct into member variables
    threadIdx(threadId),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    tt(sharedState.tt),
    networks(sharedState.networks),
    refreshTable(networks[token]) {
    clear();
}

void Search::Worker::ensure_network_replicated() {
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
}

void Search::Worker::start_searching() {

    accumulatorStack.reset();

    // --- Build Bayesian tables once per search (deterministic) ---
    // Fast-path: if the feature is disabled, avoid scanning the option map entirely.
    // --- Build (or clear) Bayesian tables once per search (deterministic) ---
    // Always re-initialize per-thread config and counters to avoid stale state
    // when BayesEnabled is toggled between searches in the same thread.
    const bool wantBayes = int(options["BayesEnabled"]) != 0;

    g_bayesCfg = BayesConfig{};  // clear any previous fields
    if (wantBayes)
    {
        g_bayesCfg = load_bayes_config(options);
        if (!bayes_has_effect(g_bayesCfg))
            g_bayesCfg.enabled = false;
        else
            g_bayesCfg.enabled = true;
    }
    else
    {
        g_bayesCfg.enabled = false;
    }

    if (g_bayesCfg.enabled)
    {
        const BayesConfig& B = g_bayesCfg;

        // Preweight capture piece values (removes mul/div from the hot path)
        for (int p = 0; p < PIECE_NB; ++p)
            g_capWeightVal[p] = (B.capWeightPct * PieceValue[p]) / 100;

        // σ(d) table
        for (int d = 0; d < 32; ++d)
            g_sigmaByDepth[d] = std::max(1, sigma_by_depth_linear(d, B));

        // p*(d) & z*(d) tables
        const int pBase = B.pstarPermille;
        const int pSh   = B.pstarShallowPermille ? B.pstarShallowPermille : pBase;
        const int pDp   = B.pstarDeepPermille ? B.pstarDeepPermille : pBase;
        int       dSh   = std::max(0, std::min(31, B.pstarShallowDepth));
        int       dDp   = std::max(1, std::min(31, B.pstarDeepDepth));
        if (dDp <= dSh)
            dDp = std::min(31, dSh + 1);
        auto lerp = [](int a, int b, int t, int T) { return a + (b - a) * t / std::max(1, T); };

        for (int d = 0; d < 32; ++d)
        {
            int p = pBase;
            if (d < dSh)
                p = pSh;
            else if (d > dDp)
                p = pDp;
            else
                p = lerp(pSh, pDp, d - dSh, dDp - dSh);
            g_pstarByDepthPermille[d] = p;
            g_zstarByDepthQ8[d]       = compute_zstar_q8(p, B.cdfMode, B.tNu);
        }

        // Reset Beta–Binomial counters
        g_bb_tried_global = g_bb_succ_global = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 3; j++)
                g_bb_tried[i][j] = g_bb_succ[i][j] = 0;
    }

    // Non-main threads go directly to iterative_deepening()
    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
    tt.new_search();

    if (rootMoves.empty())
    {
        rootMoves.emplace_back(Move::none());
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
    }
    else
    {
        threads.start_searching();  // start non-main threads
        iterative_deepening();      // main thread start searching
    }

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.
    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {}  // Busy wait for a stop or a ponder reset

    // Stop the threads if not already stopped (also raise the stop if
    // "ponderhit" just reset threads.ponder)
    threads.stop = true;

    // Wait until all threads have finished
    threads.wait_for_search_finished();

    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;
    Skill   skill =
      Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
        && rootMoves[0].pv[0] != Move::none())
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    // Send again PV info if we have a new best thread
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);

    std::string ponder;

    if (bestThread->rootMoves[0].pv.size() > 1
        || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has been
// consumed, the user stops the search, or the maximum search depth is reached.
void Search::Worker::iterative_deepening() {

    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

    Move pv[MAX_PLY + 1];

    Depth lastBestMoveDepth = 0;
    Value lastBestScore     = -VALUE_INFINITE;
    auto  lastBestPV        = std::vector{Move::none()};

    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int    delta, iterIdx                        = 0;

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    size_t multiPV = size_t(options["MultiPV"]);
    Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    // When playing with strength handicap enable MultiPV search that we will
    // use behind-the-scenes to retrieve a set of possible moves.
    if (skill.enabled())
        multiPV = std::max(multiPV, size_t(4));

    multiPV = std::min(multiPV, rootMoves.size());

    int searchAgainCounter = 0;

    lowPlyHistory.fill(97);

    // Iterative deepening loop until requested to stop or the target depth is reached
    while (++rootDepth < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth > limits.depth))
    {
        // Age out PV variability metric
        if (mainThread)
            totBestMoveChanges /= 2;

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.
        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        size_t pvFirst = 0;
        pvLast         = 0;

        if (!threads.increaseDepth)
            searchAgainCounter++;

        // MultiPV loop. We perform a full root search for each PV line
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }

            // Reset UCI info selDepth for each depth and each PV line
            selDepth = 0;
            // Reset aspiration window starting size (master behavior)
            delta     = 5 + threadIdx % 8 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 9000;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);
            // Adjust optimism based on root move's averageScore
            optimism[us]  = 137 * avg / (std::abs(avg) + 91);
            optimism[~us] = -optimism[us];

            // Start with a small aspiration window and, in the case of a fail
            // high/low, re-search with a bigger window until we don't fail
            // high/low anymore.
            int failedHighCnt = 0;
            while (true)
            {
                // Adjust the effective depth searched, but ensure at least one
                // effective increment for every four searchAgain steps (see issue #2717).
                Depth adjustedDepth =
                  std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.
                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs like Fritz 19, only start
                // at nodes > 10M (rather than depth N, which can be reached quickly)
                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > 10000000)
                    main_manager()->pv(*this, threads, tt, rootDepth);

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = alpha;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {

                    // Restore master aspiration behavior: widen both sides on fail-high
                    alpha = std::max(beta - delta, alpha);
                    beta  = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += delta / 3;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Sort the PV lines searched so far and update the GUI
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread
                && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000)
                // A thread that aborted search can have mated-in/TB-loss PV and
                // score that cannot be trusted, i.e. it can be delayed or refuted
                // if we would have had time to fully search other root-moves. Thus
                // we suppress this output and below pick a proven score/PV for this
                // thread (from the previous iteration).
                && !(threads.abortedSearch && is_loss(rootMoves[0].uciScore)))
                main_manager()->pv(*this, threads, tt, rootDepth);

            if (threads.stop)
                break;
        }

        if (!threads.stop)
            completedDepth = rootDepth;

        // We make sure not to pick an unproven mated-in score,
        // in case this thread prematurely stopped search (aborted-search).
        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score))
        {
            // Bring the last best move to the front for best thread selection.
            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](
                                                const auto& rm) { return rm == lastBestPV[0]; });
            rootMoves[0].pv    = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;
        }
        else if (rootMoves[0].pv[0] != lastBestPV[0])
        {
            lastBestPV        = rootMoves[0].pv;
            lastBestScore     = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }

        if (!mainThread)
            continue;

        // Have we found a "mate in x"?
        if (limits.mate && rootMoves[0].score == rootMoves[0].uciScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;

        // If the skill level is enabled and time is up, pick a sub-optimal best move
        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Use part of the gained time from a previous stable move for the current move
        for (auto&& th : threads)
        {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (11.325 + 2.115 * (mainThread->bestPreviousAverageScore - bestValue)
               + 0.987 * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5688, 1.5698);

            // If the bestMove is stable over several iterations, reduce time accordingly
            double k      = 0.5189;
            double center = lastBestMoveDepth + 11.57;
            timeReduction = 0.723 + 0.79 / (1.104 + std::exp(-k * (completedDepth - center)));
            double reduction =
              (1.455 + mainThread->previousTimeReduction) / (2.2375 * timeReduction);
            double bestMoveInstability = 1.04 + 1.8956 * totBestMoveChanges / threads.size();

            double totalTime =
              mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;

            // Cap used time in case of a single legal move for a better viewer experience
            if (rootMoves.size() == 1)
                totalTime = std::min(502.0, totalTime);

            auto elapsedTime = elapsed();

            if (completedDepth >= 10 && nodesEffort >= 92425 && elapsedTime > totalTime * 0.666
                && !mainThread->ponder)
                threads.stop = true;

            // Stop the search if we have exceeded the totalTime or maximum
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.503;
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return;

    mainThread->previousTimeReduction = timeReduction;

    // If the skill level is enabled, swap the best PV line with the sub-optimal one
    if (skill.enabled())
        std::swap(rootMoves[0],
                  *std::find(rootMoves.begin(), rootMoves.end(),
                             skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));
}


void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), ss);
}

void Search::Worker::do_move(
  Position& pos, const Move move, StateInfo& st, const bool givesCheck, Stack* const ss) {
    bool       capture = pos.capture_stage(move);
    DirtyPiece dp      = pos.do_move(move, st, givesCheck, &tt);
    nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);
    if (ss != nullptr)
    {
        ss->currentMove         = move;
        ss->continuationHistory = &continuationHistory[ss->inCheck][capture][dp.pc][move.to_sq()];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[dp.pc][move.to_sq()];
    }
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st) { pos.do_null_move(st, tt); }

void Search::Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
}

void Search::Worker::undo_null_move(Position& pos) { pos.undo_null_move(); }


// Reset histories, usually before a new game
void Search::Worker::clear() {
    mainHistory.fill(68);
    captureHistory.fill(-689);
    pawnHistory.fill(-1238);
    pawnCorrectionHistory.fill(5);
    minorPieceCorrectionHistory.fill(0);
    nonPawnCorrectionHistory.fill(0);

    ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(8);

    for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : continuationHistory[inCheck][c])
                for (auto& h : to)
                    h.fill(-529);

    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(2809 / 128.0 * std::log(i));

    refreshTable.clear(networks[numaAccessToken]);
}  // end of Search::Worker::clear

// Main search function for both PV and non-PV nodes
template<NodeType nodeType>
Value Search::Worker::search(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {

    constexpr bool PvNode   = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool     allNode  = !(PvNode || cutNode);

    // Dive into quiescence search when the depth reaches zero
    if (depth <= 0)
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return qsearch<nt>(pos, ss, alpha, beta);
    }

    // Limit the depth if extensions made it too large
    depth = std::min(depth, MAX_PLY - 1);

    // Check if we have an upcoming move that draws by repetition
    if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool  givesCheck, improving, priorCapture, opponentWorsening;
    bool  capture, ttCapture;
    int   priorReduction;
    Piece movedPiece;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Step 1. Initialize node
    ss->inCheck   = pos.checkers();
    priorCapture  = pos.captured_piece();
    Color us      = pos.side_to_move();
    ss->moveCount = 0;
    bestValue     = -VALUE_INFINITE;
    maxValue      = VALUE_INFINITE;

    // Check for the available remaining time
    if (is_mainthread())
        main_manager()->check_time(*this);

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (!rootNode)
    {
        // Step 2. Check for aborted search and immediate draw
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : value_draw(nodes);

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Same logic but with reversed
        // signs apply also in the opposite condition of being mated instead of giving
        // mate. In this case, return a fail-high score.
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore       = 0;
    (ss + 2)->cutoffCnt = 0;

    // Step 4. Transposition table lookup
    excludedMove                   = ss->excludedMove;
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    // Need further processing of the saved data
    ss->ttHit    = ttHit;
    ttData.move  = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // At this point, if excluded, skip straight to step 6, static eval. However,
    // to save indentation, we list the condition in all code between here and there.

    // At non-PV nodes we check for an early TT cutoff
    if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta)
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5)
        // avoid a TT cutoff if the rule50 count is high and the TT move is zeroing
        && (depth > 8 || ttData.move == Move::none() || pos.rule50_count() < 80
            || (!ttCapture && type_of(pos.moved_piece(ttData.move)) != PAWN)))
    {
        // If ttMove is quiet, update move sorting heuristics on TT hit
        if (ttData.move && ttData.value >= beta)
        {
            // Bonus for a quiet ttMove that fails high
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                       std::min(130 * depth - 71, 1043));

            // Extra penalty for early quiet moves of the previous ply
            if (prevSq != SQ_NONE && (ss - 1)->moveCount < 4 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2142);
        }

        // Partial workaround for the graph history interaction problem
        // For high rule50 counts don't produce transposition table cutoffs.
        if (pos.rule50_count() < 96)
        {
            if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) && pos.legal(ttData.move)
                && !is_decisive(ttData.value))
            {
                pos.do_move(ttData.move, st);
                Key nextPosKey                             = pos.key();
                auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
                pos.undo_move(ttData.move);

                // Check that the ttValue after the tt move would also trigger a cutoff
                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            }
            else
                return ttData.value;
        }
    }

    // Step 5. Tablebases probe
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore   wdl = Tablebases::probe_wdl(pos, &err);

            // Force check of time on the next occasion
            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                tbHits.fetch_add(1, std::memory_order_relaxed);

                int drawScore = tbConfig.useRule50 ? 1 : 0;

                Value tbValue = VALUE_TB - ss->ply;

                // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to score
                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                   std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                   tt.generation());

                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }

    // Step 6. Static evaluation of the position
    Value      unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue      = correction_value(*this, pos, ss);
    if (ss->inCheck)
    {
        // Skip early pruning when in check
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
        // (goto removed) fallthrough to moves_loop guarded by if (!ss->inCheck)
    }
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // ttValue can be used as a better position evaluation
        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // Static evaluation is saved as it was before adjustment by correction history
        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation());
    }

    // Use static evaluation difference to improve quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        int bonus = std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -2023, 1563) + 583;
        mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 944 / 1024;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << bonus * 1438 / 1024;
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    improving         = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    if (priorReduction >= 3 && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 173)
        depth--;

    // Step 7. Razoring
    // If eval is really low, skip search entirely and return the qsearch value.
    // For PvNodes, we must have a guard against mates being returned.
    if (!PvNode && eval < alpha - 514 - 294 * depth * depth)
        return qsearch<NonPV>(pos, ss, alpha, beta);

    // Step 8. Futility pruning: child node
    // The depth condition is important for mate finding.
    {
        auto futility_margin = [&](Depth d) {
            Value futilityMult = 91 - 21 * !ss->ttHit;

            return futilityMult * d                                //
                 - 2094 * improving * futilityMult / 1024          //
                 - 1324 * opponentWorsening * futilityMult / 4096  //
                 + (ss - 1)->statScore / 331                       //
                 + std::abs(correctionValue) / 158105;
        };

        if (!ss->ttPv && depth < 14 && eval - futility_margin(depth) >= beta && eval >= beta
            && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
            return (2 * beta + eval) / 3;
    }

    // Step 9. Null move search with verification search
    if (cutNode && ss->staticEval >= beta - 18 * depth + 390 && !excludedMove
        && pos.non_pawn_material(us) && ss->ply >= nmpMinPly && !is_loss(beta))
    {
        assert((ss - 1)->currentMove != Move::null());

        // Null move dynamic reduction based on depth
        Depth R = 6 + depth / 3;

        ss->currentMove                   = Move::null();
        ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

        do_null_move(pos, st);

        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        if (nullValue >= beta && !is_win(nullValue))
        {
            if (nmpMinPly || depth < 16)
                return nullValue;

            assert(!nmpMinPly);  // Recursive verification is not allowed

            // Do verification search at high depths, with null move pruning disabled
            // until ply exceeds nmpMinPly.
            nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    improving |= ss->staticEval >= beta;

    // Step 10. Internal iterative reductions
    // At sufficient depth, reduce depth for PV/Cut nodes without a TTMove.
    // (*Scaler) Especially if they make IIR less aggressive.
    if (!allNode && depth >= 6 && !ttData.move && priorReduction <= 3)
        depth--;

    // Track whether TT evidence suggests skipping the Bayes gate (but still
    // allow the standard ProbCut verification). Function-scoped so Step-12 can read it.
    bool  bypassGateDueToTT = false;
    Depth probCutDepth      = Depth(0);
    // Function-scoped so Step-11/12 can share these
    probCutBeta = beta;  // will be set below


    // Skip all ProbCut variants when in check (align with master)
    if (!ss->inCheck)
    {
        // Step 11. ProbCut
        // If we have a good enough capture (or queen promotion) and a reduced search
        // returns a value much above beta, we can (almost) safely prune the previous move.
        probCutBeta = beta + 224 - 64 * improving;
        if (g_bayesCfg.enabled)
        {
            probCutBeta =
              beta
              + Value(g_bayesCfg.probCutBetaBaseCp - g_bayesCfg.probCutBetaImproveCp * improving);
        }


        if (depth >= 3
            && !is_decisive(beta)
            // If value from transposition table is lower than probCutBeta, don't attempt
            // probCut there
            && !(is_valid(ttData.value) && ttData.value < probCutBeta))
        {
            if (probCutBeta <= beta)
                probCutBeta = beta + 1;
            assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);
            // Runs only when !ss->inCheck; hoisted here to live next to the Bayes gate for locality.
            const int pcd =
              g_bayesCfg.enabled
                ? std::clamp<int>(int(depth) - g_bayesCfg.probCutDepthOffset
                                    - int(ss->staticEval - beta) / 306,
                                  0, int(depth))
                : std::clamp<int>(int(depth) - 5 - int(ss->staticEval - beta) / 306, 0, int(depth));
            probCutDepth = Depth(pcd);

            // If TT shows we're comfortably under β (UB) or comfortably over β (LB),
            // bypass only the Bayes gate, not the standard ProbCut verification
            // (preserves master pruning).
            if (g_bayesCfg.enabled)
            {

                bypassGateDueToTT =
                  (is_valid(ttData.value) && ttHit && !is_decisive(ttData.value)
                   && std::abs(int(ttData.value)) < int(VALUE_TB)
                   && ttData.depth >= g_bayesCfg.ttLBMinDepth && probCutDepth >= 2 && depth >= 6
                   && (((ttData.bound & BOUND_UPPER)
                        && (ttData.value + Value(g_bayesCfg.ttNearCp)) < probCutBeta)
                       || ((ttData.bound & BOUND_LOWER)
                           && (ttData.value - Value(g_bayesCfg.ttNearCp)) >= probCutBeta)));
            }
            else
            {
                bypassGateDueToTT = false;  // Bayes disabled → no TT bypass
                // Bayes disabled → keep standard ProbCut; no special per-node verification budget here.
            }
        }

        // Step 12. A small ProbCut idea
        if (g_bayesCfg.enabled)
        {
            const int spc = g_bayesCfg.smallProbCutCp;
            if (spc > 0 && (ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4)
            {
                probCutBeta = beta + spc;
                if (!is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value)
                    && ttData.value >= probCutBeta)
                    return probCutBeta;
            }
            // [Hoisted] Bayesian precompute + ProbCut verification loop
            // --- Bayesian node precompute (cheap, integer-only) ---
            const BayesConfig& B        = g_bayesCfg;
            bool               tb_avoid = false;
            if (B.tbGuard)
            {
                if (tbConfig.rootInTB)
                    tb_avoid = true;
                else if (tbConfig.cardinality)
                {
                    int _tbPieces = pos.count<ALL_PIECES>();
                    if (_tbPieces <= tbConfig.cardinality)
                        tb_avoid = true;
                }
            }
            bool runBayes = B.enabled && !PvNode && !bypassGateDueToTT
                         && probCutDepth >= B.gateMinDepth && !tb_avoid
                         && !(B.noGateInCheck && ss->inCheck) && (!B.gateCutOnly || cutNode)
                         && !(B.noGateAtDepth > 0 && ss->ply <= B.noGateAtDepth)
                         && !(B.noGateRule50 > 0 && pos.rule50_count() >= B.noGateRule50);

            int zStarQ8Node = 0, sigmaNode = 1, thrQ8Node = 0, thrQ8NodeBest = 0;
            int eviReqCp = -1, muNodeBiasCp = 0;

            int  thrMtcStep = 0, thrMtcCap = 0;  // Multiple-Tests Correction (Q8·σ units)
            bool ttNearLBDone = false,
                 ttNearUBDone = false;  // track near-boosts separately for LB and UB
            int verifCount = 0, prunedByGate = 0, thrMtcAcc = 0;
            // Defensive: keep the accumulator inside bounds from the start

            if (runBayes)
            {
                const int  rd           = std::clamp<int>(int(probCutDepth), 0, 31);
                int        sigma        = g_sigmaByDepth[rd];
                const bool shallowGuard = (rd <= 3);
                const bool skipBayesNode =
                  ((B.noGateInCheck || shallowGuard) && ss->inCheck)
                  || ((B.noGateRule50 > 0) ? (pos.rule50_count() >= B.noGateRule50)
                                           : (pos.rule50_count() > 0));
                if (skipBayesNode)
                {
                    runBayes = false;
                }
                else
                {
                    // Phase-aware σ
                    const int npmBoth = pos.non_pawn_material(WHITE) + pos.non_pawn_material(BLACK);
                    const int phaseScale =
                      (npmBoth <= 2400 ? B.sigmaEndPct : (npmBoth <= 4800 ? B.sigmaMidPct : 100));
                    sigma = std::max(1, sigma * phaseScale / 100);

                    // TT confidence & near-bound
                    const bool freshTT_local = ttHit;
                    const bool ttValOK_local = is_valid(ttData.value) && !is_decisive(ttData.value)
                                            && std::abs(int(ttData.value)) < int(VALUE_TB);
                    const bool ttLB = freshTT_local && ttValOK_local && (ttData.bound & BOUND_LOWER)
                                   && ttData.depth >= B.ttLBMinDepth;
                    const bool ttUB = freshTT_local && ttValOK_local && (ttData.bound & BOUND_UPPER)
                                   && ttData.depth >= B.ttLBMinDepth;
                    // If TT says we're comfortably below beta, disable only the Bayes gate.

                    // σ shaving when TT info is strong/near-bound
                    if (ttLB)
                    {
                        sigma = std::max(1, sigma - B.ttBoostCp);
                        if (B.ttNearBoostCp > 0)
                        {
                            const int dv = int(ttData.value);
                            const int pb = int(probCutBeta);
                            if (!ttNearLBDone && std::abs(dv - pb) <= B.ttNearCp)
                            {
                                ttNearLBDone = true;
                                sigma        = std::max(1, sigma - B.ttNearBoostCp);
                            }
                        }
                    }
                    if (ttUB)
                    {
                        sigma = std::max(1, sigma - B.ttBoostCp);
                        if (B.ttNearBoostCp > 0)
                        {
                            const int dv = int(ttData.value);
                            const int pb = int(probCutBeta);
                            if (!ttNearUBDone && std::abs(dv - pb) <= B.ttNearCp)
                            {
                                ttNearUBDone = true;
                                sigma        = std::max(1, sigma - B.ttNearBoostCp / 2);
                            }
                        }
                    }  // Volatility-aware σ inflation
                    if (B.sigmaVolAddCp > 0 && B.sigmaVolThreshCp > 0 && ss->ply > 0
                        && is_valid((ss - 1)->staticEval))
                    {
                        const int prevEval = (ss - 1)->staticEval;
                        if (std::abs(int(ss->staticEval) - prevEval) >= B.sigmaVolThreshCp)
                            sigma += B.sigmaVolAddCp;
                    }

                    if (B.sigmaFloorCp > 0)
                        sigma = std::max(sigma, B.sigmaFloorCp);
                    if (B.sigmaCeilCp > 0)
                        sigma = std::min(sigma, B.sigmaCeilCp);
                    sigmaNode = sigma;

                    // Base z* from p*(rd) with biases
                    zStarQ8Node = std::max(0, g_zstarByDepthQ8[rd]);
                    if (ttLB && B.zTTBiasQ8 > 0)
                        zStarQ8Node -= std::min(zStarQ8Node, B.zTTBiasQ8);
                    if (ttUB && B.zTTUBiasQ8 > 0)
                        zStarQ8Node += B.zTTUBiasQ8;

                    if (B.zMidBiasQ8 || B.zEndBiasQ8)
                    {
                        const int zPhaseBias =
                          (npmBoth <= 2400 ? B.zEndBiasQ8 : (npmBoth <= 4800 ? B.zMidBiasQ8 : 0));
                        if (zPhaseBias > 0)
                            zStarQ8Node -= std::min(zStarQ8Node, zPhaseBias);
                    }

                    if (ss->ply >= 2)
                    {
                        const bool improving2 = int(ss->staticEval) >= (ss - 2)->staticEval;
                        if (improving2 && B.zImprovingBiasQ8 > 0)
                            zStarQ8Node -= std::min(zStarQ8Node, B.zImprovingBiasQ8);
                        else if (!improving2 && B.zNotImprovingBiasQ8 > 0)
                            zStarQ8Node += B.zNotImprovingBiasQ8;
                    }
                    if (ss->ttPv && B.zPvBiasQ8 > 0)
                        zStarQ8Node += B.zPvBiasQ8;
                    if (B.zTightBiasQ8 > 0 && B.tightWindowCp > 0)
                    {
                        const bool tightOK =
                          (!B.tightOnPVOnly || ss->ttPv) && (beta - alpha) <= B.tightWindowCp;
                        if (tightOK)
                            zStarQ8Node += B.zTightBiasQ8;
                    }
                    if (B.zStarMinQ8 > 0)
                        zStarQ8Node = std::max(zStarQ8Node, B.zStarMinQ8);

                    // Beta–Binomial posterior → small z* shave where success rate ≥ prior (bucket+global mix)
                    if (B.bbEnabled && B.bbGainQ8 > 0 && B.bbMaxBiasQ8 > 0)
                    {
                        // Bucket by reduced ProbCut depth and phase; use per-thread counters.
                        const int      db = bb_depth_bucket(probCutDepth);
                        const int      pb = bb_phase_bucket(pos.non_pawn_material(WHITE)
                                                            + pos.non_pawn_material(BLACK));
                        const uint32_t tB = g_bb_tried[db][pb];
                        const uint32_t sB = g_bb_succ[db][pb];
                        const uint32_t tG = g_bb_tried_global;
                        const uint32_t sG = g_bb_succ_global;

                        // Conjugate priors → posterior means in permille
                        const int aB    = B.bbAlpha0 + int(sB);
                        const int bB    = B.bbBeta0 + std::max(0, int(tB) - int(sB));
                        const int aG    = B.bbAlpha0 + int(sG);
                        const int bG    = B.bbBeta0 + std::max(0, int(tG) - int(sG));
                        const int postB = int((1000LL * aB) / std::max(1, aB + bB));
                        const int postG = int((1000LL * aG) / std::max(1, aG + bG));

                        // Trials‑aware mixing: ramp bucket weight up to bbMixPermille as trials→bbMinTrials
                        const int minT   = std::max(1, B.bbMinTrials);
                        const int trials = std::min<int>(int(tB), minT);
                        int       lambda = (B.bbMixPermille * trials) / minT;  // 0..1000
                        lambda           = std::clamp(lambda, 0, 1000);

                        const int postEff = (lambda * postB + (1000 - lambda) * postG) / 1000;
                        const int deltaP =
                          std::max(0, postEff - g_pstarByDepthPermille[rd]);  // permille

                        int biasQ8  = (B.bbGainQ8 * deltaP) / 10;  // permille → percent (10 ‰)
                        biasQ8      = std::min(biasQ8, B.bbMaxBiasQ8);
                        zStarQ8Node = std::max(0, zStarQ8Node - biasQ8);
                    }
                    // Hoist thresholds
                    int zBest = zStarQ8Node;
                    if (B.zTTBestBiasQ8 > 0)
                        zBest = std::max(0, zBest - B.zTTBestBiasQ8);
                    thrQ8Node     = mul_q8_cp_clamped(zStarQ8Node, std::max(1, sigmaNode));
                    thrQ8NodeBest = mul_q8_cp_clamped(zBest, std::max(1, sigmaNode));
                    // EVI per-depth (optional)
                    if (B.eviMinCp > 0)
                    {
                        // Guard against p*(d) being set to 0 via UCI; avoid division by zero.
                        int pPerm = g_pstarByDepthPermille[rd];
                        pPerm     = std::max(1, pPerm);
                        // EVI scaling: p* is in permille; clamp denominator to 700 to avoid tiny p*
                        eviReqCp =
                          (B.eviMinCp * 1000 + std::max(700, pPerm) - 1) / std::max(700, pPerm);
                        eviReqCp += std::max(0, sigmaNode / 4);
                        eviReqCp = std::min(eviReqCp, 5000);  // ceil
                    }

                    // TT→μ recenter (node-level)
                    if (B.muTTAdjDiv > 0 && is_valid(ttData.value) && !is_decisive(ttData.value))
                    {
                        int adj = (int(ttData.value) - int(ss->staticEval)) / B.muTTAdjDiv;
                        if (B.muTTAdjCapCp > 0)
                        {
                            if (adj > B.muTTAdjCapCp)
                                adj = B.muTTAdjCapCp;
                            if (adj < -B.muTTAdjCapCp)
                                adj = -B.muTTAdjCapCp;
                        }
                        muNodeBiasCp = adj;
                    }

                    // MTC step/cap (converted to Q8·σ once)
                    if (B.zMtcStepQ8 > 0)
                    {
                        thrMtcStep = B.zMtcStepQ8 * sigmaNode;
                        thrMtcCap  = (B.zMtcMaxQ8 > 0 ? B.zMtcMaxQ8 * sigmaNode : 4 * sigmaNode);
                        // Keep the accumulator in range from the start.
                        // Initialize accumulator at 0; increment strictly after each verification attempt
                        thrMtcAcc = 0;
                    }

                    MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
                    while ((move = mp.next_move()) != Move::none())
                    {

                        // Enforce hard cap on total verifications per node (not just gated ones)
                        // Note: We do not enforce a per-node verification budget here; we verify all candidate captures per standard ProbCut.andidates normally but disable the Bayes gate for this move.
                        const bool gateThisMove =
                          runBayes && !(B.maxVerifPerNode > 0 && verifCount >= B.maxVerifPerNode);

                        // If we already verified enough candidates in this node, stop early
                        assert(move.is_ok());

                        if (move == excludedMove || !pos.legal(move))
                            continue;

                        assert(pos.capture_stage(move));

                        // --- Bayesian ProbCut candidate gate (cheap, integer-only) ---
                        if (gateThisMove)
                        {

                            // Respect optional budgets: if budget exhausted, skip only the Bayesian gate
                            // and fall back to normal ProbCut verification. Also disable the gate when
                            // a strong TT upper bound suggests we're below β.
                            bool gateEligible =
                              !(B.maxGatedCaptures > 0 && prunedByGate >= B.maxGatedCaptures);
                            gateEligible = gateEligible && !bypassGateDueToTT;

                            // If TT says we're comfortably below beta, bypass the Bayes gate (always verify)
                            // (We already computed bypassGateDueToTT above at node scope.)
                            // Determine if this move is eligible for gating (else proceed normally)

                            if (B.minCapturedValueCp > 0)
                            {
                                Piece capPreMin = CapturedPiecePre(pos, move);
                                if (move.type_of() == EN_PASSANT)
                                    capPreMin = make_piece(~pos.side_to_move(), PAWN);
                                if (capPreMin == NO_PIECE
                                    || PieceValue[capPreMin] < B.minCapturedValueCp)
                                    gateEligible = false;
                            }

                            // Optional SEE pre-gate; also feeds μ boost
                            const bool seeNonNeg = pos.see_ge(move, 0);  // used for μ boost only
                            // SEE gate: require SEE >= -seeMargin (negative margin allows small speculative sacs).
                            if (B.seeMarginCp > 0 && !seeNonNeg)
                            {
                                if (!pos.see_ge(move, -B.seeMarginCp))
                                {
                                    gateEligible =
                                      false;  // disable Bayes gate only; ProbCut verification still allowed
                                }
                            }
                            // Fast path: if margin==0 and SEE<0, skip Bayes gate entirely (keep standard verification)
                            if (B.seeMarginCp == 0 && !seeNonNeg)
                            {
                                gateEligible = false;
                            }
                            if (gateEligible)
                            {
                                // Prepare μ terms
                                Piece movedPieceLocal  = pos.moved_piece(move);
                                Piece capturedPiecePre = CapturedPiecePre(pos, move);
                                if (move.type_of()
                                    == EN_PASSANT)  // pre-move EP fix: captured pawn is behind 'to'
                                    capturedPiecePre = make_piece(~pos.side_to_move(), PAWN);

                                // Capture-history term only if there is a captured piece (matches master style)
                                int captHistScore = 0;
                                if (capturedPiecePre != NO_PIECE)
                                {
                                    const PieceType cpt = type_of(capturedPiecePre);
                                    captHistScore =
                                      captureHistory[movedPieceLocal][move.to_sq()][cpt];
                                }

                                int classBiasCp = 0;
                                if (move.type_of() == PROMOTION)
                                    classBiasCp += B.muPromoBiasCp;
                                const PieceType capPT = (capturedPiecePre != NO_PIECE)
                                                        ? type_of(capturedPiecePre)
                                                        : NO_PIECE_TYPE;
                                if (capPT == QUEEN || capPT == ROOK)
                                    classBiasCp += B.muMajorCapBiasCp;

                                const int capWeightPre = (capturedPiecePre != NO_PIECE)
                                                         ? g_capWeightVal[capturedPiecePre]
                                                         : 0;

                                int promoDelta = 0;
                                if (move.type_of() == PROMOTION)
                                {
                                    PieceType pt = move.promotion_type();
                                    promoDelta   = PieceValue[make_piece(pos.side_to_move(), pt)]
                                               - PieceValue[make_piece(pos.side_to_move(), PAWN)];
                                    promoDelta = std::clamp(promoDelta, -2000, 2000);
                                }
                                const int muBiasNode = classBiasCp + muNodeBiasCp
                                                     + (seeNonNeg ? B.muSEEBoostCp : 0)
                                                     + promoDelta;

                                // Threshold selection (TT-best + progressive MTC)
                                const bool isTTBest   = (ttData.move && move == ttData.move);
                                int        thrForMove = (isTTBest ? thrQ8NodeBest : thrQ8Node);
                                // If this is a queen promotion, give a small extra allowance even if not TT-best
                                if (move.type_of() == PROMOTION && move.promotion_type() == QUEEN)
                                {
                                    // reduce threshold by ~0.125σ
                                    thrForMove = std::max(0, thrForMove - 32 * sigmaNode);
                                }
                                if (thrMtcStep > 0)
                                {
                                    const int add =
                                      (thrMtcCap > 0 ? std::min(thrMtcAcc, thrMtcCap) : thrMtcAcc);
                                    thrForMove = std::min<int>(thrForMove + add, int(Q8_RAIL));
                                }
                                // Keep threshold within the same numeric domain as diffQ8
                                thrForMove = std::min<int>(thrForMove, int(Q8_RAIL));
                                // MTC accumulator increments only when we attempt verification (gatePass==true)
                                const bool gatePass =
                                  bayes_probcut_gate(ss->staticEval,  // Value staticEval
                                                     probCutBeta,     // Value probCutBeta
                                                     captHistScore,   // int captHist
                                                     capWeightPre,    // int capWeightPre
                                                     muBiasNode,      // int muBiasNode
                                                     thrForMove,  // int thrQ8Node (z*·σ in Q8·cp)
                                                     eviReqCp,    // int eviReqCp (≥0 => enforce)
                                                     B);
                                if (gatePass && thrMtcStep > 0)
                                {
                                    // Post-attempt MTC: bump after deciding to verify this move
                                    thrMtcAcc =
                                      std::clamp(thrMtcAcc + thrMtcStep, 0,
                                                 (thrMtcCap > 0 ? thrMtcCap : int(Q8_RAIL)));
                                }
                                // const BayesConfig&

                                if (!gatePass)
                                {
                                    ++prunedByGate;
                                    continue;
                                }  // Bayes gate rejected → skip ProbCut verification

                                // (Duplicate BB trial removed; we count once below.)
                            }
                        }


                        // Record a BB trial only when we actually verify (not when the Bayes gate prunes)
                        if (B.bbEnabled)
                        {
                            const int db = bb_depth_bucket(probCutDepth);
                            const int pb = bb_phase_bucket(pos.non_pawn_material(WHITE)
                                                           + pos.non_pawn_material(BLACK));
                            sat_inc(g_bb_tried[db][pb]);
                            sat_inc(g_bb_tried_global);
                            maybe_age_bb();
                        }

                        if (runBayes && B.maxVerifPerNode > 0)
                            ++verifCount;
                        do_move(pos, move, st, ss);
                        // Perform a preliminary qsearch to verify that the move holds
                        value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

                        // If the qsearch held, perform the regular search
                        if (value >= probCutBeta && probCutDepth > 0)
                            value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1,
                                                   probCutDepth, !cutNode);

                        undo_move(pos, move);

                        if (value >= probCutBeta)
                        {
                            // Record BB success (exclude decisive/TB-like results to avoid biasing posterior)
                            if (B.bbEnabled && !is_decisive(value)
                                && std::abs(int(value)) < int(VALUE_TB))
                            {
                                const int db = bb_depth_bucket(probCutDepth);
                                const int pb = bb_phase_bucket(pos.non_pawn_material(WHITE)
                                                               + pos.non_pawn_material(BLACK));
                                sat_inc(g_bb_succ[db][pb]);
                                sat_inc(g_bb_succ_global);
                            }
                            // Save ProbCut data into transposition table
                            ttWriter.write(
                              posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                              probCutDepth + 1, move,
                              (is_valid(unadjustedStaticEval) ? unadjustedStaticEval : VALUE_NONE),
                              tt.generation());

                            // Master behavior: for non-decisive scores, return a softened bound
                            // to avoid aspiration instability; for decisive scores, keep the raw value.
                            if (!is_decisive(value))
                                return value - (probCutBeta - beta);
                            else
                                return value;  // decisive → keep raw value and stop

                            // (Standard ProbCut verification path…)
                            // If verification cuts, return; otherwise continue with next candidate.
                            // (Existing cut/TT write code remains unchanged.)
                        }
                    }
                }
            }  // end: skip ProbCut when in check
            if (!g_bayesCfg.enabled)
            {


                // Step 12. A small Probcut idea
                probCutBeta = beta + 418;
                if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4
                    && is_valid(ttData.value) && !is_decisive(beta) && !is_decisive(ttData.value)
                    && ttData.value >= probCutBeta)
                    return probCutBeta;
            }
        }
        const PieceToHistory* contHist[] = {
          (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
          (ss - 3)->continuationHistory, (ss - 4)->continuationHistory,
          (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

        MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory,
                      contHist, &pawnHistory, ss->ply);

        value = bestValue;

        int moveCount = 0;
        // Step 13. Loop through all pseudo-legal moves until no moves remain
        // or a beta cutoff occurs.
        while ((move = mp.next_move()) != Move::none())
        {

            // Enforce hard cap on total verifications per node (not just gated ones)
            assert(move.is_ok());

            if (move == excludedMove)
                continue;

            // Check for legality
            if (!pos.legal(move))
                continue;
            if (rootNode
                && !std::count(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast, move))
                continue;

            ss->moveCount = ++moveCount;

            if (rootNode && is_mainthread() && nodes > 10000000)
            {
                main_manager()->updates.onIter(
                  {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
            }
            if (PvNode)
                (ss + 1)->pv = nullptr;

            extension  = 0;
            capture    = pos.capture_stage(move);
            movedPiece = pos.moved_piece(move);
            givesCheck = pos.gives_check(move);

            // Calculate new depth for this move
            newDepth = depth - 1;

            int delta = beta - alpha;

            Depth r = reduction(improving, depth, moveCount, delta);

            // Increase reduction for ttPv nodes (*Scaler)
            // Smaller or even negative value is better for short time controls
            // Bigger value is better for long time controls
            if (ss->ttPv)
                r += 946;

            // Step 14. Pruning at shallow depth.
            // Depth conditions are important for mate finding.
            if (!rootNode && pos.non_pawn_material(us) && !is_loss(bestValue))
            {
                // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
                if (moveCount >= (3 + depth * depth) / (2 - improving))
                    mp.skip_quiet_moves();

                // Reduced depth of the next LMR search
                int lmrDepth = newDepth - r / 1024;

                if (capture || givesCheck)
                {
                    Piece capturedPiece =
                      g_bayesCfg.enabled ? CapturedPiecePre(pos, move) : pos.piece_on(move.to_sq());
                    int captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                    // Futility pruning for captures
                    if (!givesCheck && lmrDepth < 7)
                    {
                        Value futilityValue = ss->staticEval + 231 + 211 * lmrDepth
                                            + PieceValue[capturedPiece] + 130 * captHist / 1024;

                        if (futilityValue <= alpha)
                            continue;
                    }

                    // SEE based pruning for captures and checks
                    // Avoid pruning sacrifices of our last piece for stalemate
                    int margin = std::max(157 * depth + captHist / 29, 0);
                    if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                        && !pos.see_ge(move, -margin))
                        continue;
                }
                else
                {
                    int history = (*contHist[0])[movedPiece][move.to_sq()]
                                + (*contHist[1])[movedPiece][move.to_sq()]
                                + pawnHistory[pawn_history_index(pos)][movedPiece][move.to_sq()];

                    // Continuation history based pruning
                    if (history < -4312 * depth)
                        continue;

                    history += 76 * mainHistory[us][move.from_to()] / 32;

                    // (*Scaler): Generally, a lower divisor scales well
                    lmrDepth += history / 3220;

                    Value futilityValue = ss->staticEval + 47 + 171 * !bestMove + 134 * lmrDepth
                                        + 90 * (ss->staticEval > alpha);

                    // Futility pruning: parent node
                    // (*Scaler): Generally, more frequent futility pruning
                    // scales well with respect to time and threads
                    if (!ss->inCheck && lmrDepth < 11 && futilityValue <= alpha)
                    {
                        if (bestValue <= futilityValue && !is_decisive(bestValue)
                            && !is_win(futilityValue))
                            bestValue = futilityValue;
                        continue;
                    }

                    lmrDepth = std::max(lmrDepth, 0);

                    // Prune moves with negative SEE
                    if (!pos.see_ge(move, -27 * lmrDepth * lmrDepth))
                        continue;
                }
            }

            // Step 15. Extensions
            // Singular extension search. If all moves but one
            // fail low on a search of (alpha-s, beta-s), and just one fails high on
            // (alpha, beta), then that move is singular and should be extended. To
            // verify this we do a reduced search on the position excluding the ttMove
            // and if the result is lower than ttValue minus a margin, then we will
            // extend the ttMove. Recursive singular search is avoided.

            // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
            // and lower extension margins scale well.

            if (!rootNode && move == ttData.move && !excludedMove && depth >= 6 + ss->ttPv
                && is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 3)
            {
                Value singularBeta  = ttData.value - (56 + 81 * (ss->ttPv && !PvNode)) * depth / 60;
                Depth singularDepth = newDepth / 2;

                ss->excludedMove = move;
                value =
                  search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
                ss->excludedMove = Move::none();

                if (value < singularBeta)
                {
                    int corrValAdj   = std::abs(correctionValue) / 229958;
                    int doubleMargin = -4 + 198 * PvNode - 212 * !ttCapture - corrValAdj
                                     - 921 * ttMoveHistory / 127649 - (ss->ply > rootDepth) * 45;
                    int tripleMargin = 76 + 308 * PvNode - 250 * !ttCapture + 92 * ss->ttPv
                                     - corrValAdj - (ss->ply * 2 > rootDepth * 3) * 52;

                    extension = 1 + (value < singularBeta - doubleMargin)
                              + (value < singularBeta - tripleMargin);

                    depth++;
                }

                // Multi-cut pruning
                // Our ttMove is assumed to fail high based on the bound of the TT entry,
                // and if after excluding the ttMove with a reduced search we fail high
                // over the original beta, we assume this expected cut-node is not
                // singular (multiple moves fail high), and we can prune the whole
                // subtree by returning a softbound.
                else if (value >= beta && !is_decisive(value))
                {
                    ttMoveHistory << std::max(-400 - 100 * depth, -4000);
                    return value;
                }

                // Negative extensions
                // If other moves failed high over (ttValue - margin) without the
                // ttMove on a reduced search, but we cannot do multi-cut because
                // (ttValue - margin) is lower than the original beta, we do not know
                // if the ttMove is singular or can do a multi-cut, so we reduce the
                // ttMove in favor of other moves based on some conditions:

                // If the ttMove is assumed to fail high over current beta
                else if (ttData.value >= beta)
                    extension = -3;

                // If we are on a cutNode but the ttMove is not assumed to fail high
                // over current beta
                else if (cutNode)
                    extension = -2;
            }

            // Step 16. Make the move
            do_move(pos, move, st, givesCheck, ss);

            // Add extension to new depth
            newDepth += extension;
            uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

            // Decrease reduction for PvNodes (*Scaler)
            if (ss->ttPv)
                r -= 2618 + PvNode * 991 + (ttData.value > alpha) * 903
                   + (ttData.depth >= depth) * (978 + cutNode * 1051);

            // These reduction adjustments have no proven non-linear scaling

            r += 843;  // Base reduction offset to compensate for other tweaks
            r -= moveCount * 66;
            r -= std::abs(correctionValue) / 30450;

            // Increase reduction for cut nodes
            if (cutNode)
                r += 3094 + 1056 * !ttData.move;

            // Increase reduction if ttMove is a capture
            if (ttCapture)
                r += 1415;

            // Increase reduction if next ply has a lot of fail high
            if ((ss + 1)->cutoffCnt > 2)
                r += 1051 + allNode * 814;

            // For first picked move (ttMove) reduce reduction
            if (move == ttData.move)
                r -= 2018;

            if (capture)
                ss->statScore =
                  803 * int(PieceValue[pos.captured_piece()]) / 128
                  + captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())];
            else
                ss->statScore = 2 * mainHistory[us][move.from_to()]
                              + (*contHist[0])[movedPiece][move.to_sq()]
                              + (*contHist[1])[movedPiece][move.to_sq()];

            // Decrease/increase reduction for moves with a good/bad history
            r -= ss->statScore * 794 / 8192;

            // Step 17. Late moves reduction / extension (LMR)
            if (depth >= 2 && moveCount > 1)
            {
                // In general we want to cap the LMR depth search at newDepth, but when
                // reduction is negative, we allow this move a limited search extension
                // beyond the first move depth.
                // To prevent problems when the max value is less than the min value,
                // std::clamp has been replaced by a more robust implementation.
                Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 2)) + PvNode;

                ss->reduction = newDepth - d;
                value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
                ss->reduction = 0;

                // Do a full-depth search when reduced LMR search fails high
                // (*Scaler) Usually doing more shallower searches
                // doesn't scale well to longer TCs
                if (value > alpha)
                {
                    // Adjust full-depth search based on LMR results - if the result was
                    // good enough search deeper, if it was bad enough search shallower.
                    const bool doDeeperSearch =
                      d < newDepth && value > (bestValue + 43 + 2 * newDepth);
                    const bool doShallowerSearch = value < bestValue + 9;

                    newDepth += doDeeperSearch - doShallowerSearch;

                    if (newDepth > d)
                        value =
                          -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                    // Post LMR continuation history updates
                    update_continuation_histories(ss, movedPiece, move.to_sq(), 1365);
                }
            }

            // Step 18. Full-depth search when LMR is skipped
            else if (!PvNode || moveCount > 1)
            {
                // Increase reduction if ttMove is not present
                if (!ttData.move)
                    r += 1118;

                // Note that if expected reduction is high, we reduce search depth here
                value =
                  -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                 newDepth - (r > 3212) - (r > 4784 && newDepth > 2), !cutNode);
            }

            // For PV nodes only, do a full PV search on the first move or after a fail high,
            // otherwise let the parent node fail low with value <= alpha and try another move.
            if (PvNode && (moveCount == 1 || value > alpha))
            {
                (ss + 1)->pv    = pv;
                (ss + 1)->pv[0] = Move::none();

                // Extend move from transposition table if we are about to dive into qsearch.
                if (move == ttData.move && ttData.depth > 1 && rootDepth > 8)
                    newDepth = std::max(newDepth, 1);

                value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
            }

            // Step 19. Undo move
            undo_move(pos, move);

            assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

            // Step 20. Check for a new best move
            // Finished searching the move. If a stop occurred, the return value of
            // the search cannot be trusted, and we return immediately without updating
            // best move, principal variation nor transposition table.
            if (threads.stop.load(std::memory_order_relaxed))
                return VALUE_ZERO;

            if (rootNode)
            {
                RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

                rm.effort += nodes - nodeCount;

                rm.averageScore =
                  rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

                rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                      ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                      : value * std::abs(value);

                // PV move or new best move?
                if (moveCount == 1 || value > alpha)
                {
                    rm.score = rm.uciScore = value;
                    rm.selDepth            = selDepth;
                    rm.scoreLowerbound = rm.scoreUpperbound = false;

                    if (value >= beta)
                    {
                        rm.scoreLowerbound = true;
                        rm.uciScore        = beta;
                    }
                    else if (value <= alpha)
                    {
                        rm.scoreUpperbound = true;
                        rm.uciScore        = alpha;
                    }

                    rm.pv.resize(1);

                    assert((ss + 1)->pv);

                    for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                        rm.pv.push_back(*m);

                    // We record how often the best move has been changed in each iteration.
                    // This information is used for time management. In MultiPV mode,
                    // we must take care to only do this for the first PV line.
                    if (moveCount > 1 && !pvIdx)
                        ++bestMoveChanges;
                }
                else
                    // All other moves but the PV, are set to the lowest value: this
                    // is not a problem when sorting because the sort is stable and the
                    // move position in the list is preserved - just the PV is pushed up.
                    rm.score = -VALUE_INFINITE;
            }

            // In case we have an alternative move equal in eval to the current bestmove,
            // promote it to bestmove by pretending it just exceeds alpha (but not beta).
            int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                       && !is_win(std::abs(value) + 1));

            if (value + inc > bestValue)
            {
                bestValue = value;

                if (value + inc > alpha)
                {
                    bestMove = move;

                    if (PvNode && !rootNode)  // Update pv even in fail-high case
                        update_pv(ss->pv, move, (ss + 1)->pv);

                    if (value >= beta)
                    {
                        // (*Scaler) Especially if they make cutoffCnt increment more often.
                        ss->cutoffCnt += (extension < 2) || PvNode;
                        assert(value >= beta);  // Fail high
                        break;
                    }

                    // Reduce other moves if we have found at least one score improvement
                    if (depth > 2 && depth < 14 && !is_decisive(value))
                        depth -= 2;

                    assert(depth > 0);
                    alpha = value;  // Update alpha! Always alpha < beta
                }
            }

            // If the move is worse than some previously searched move,
            // remember it, to update its stats later.
            if (move != bestMove)
            {
                if (capture)
                {
                    if (capturesSearched.size() < 32)
                        capturesSearched.push_back(move);
                }
                else
                {
                    if (quietsSearched.size() < 32)
                        quietsSearched.push_back(move);
                }
            }

            // Step 21. Check for mate and stalemate
            // All legal moves have been searched and if there are no legal moves, it
            // must be a mate or a stalemate. If we are in a singular extension search then
            // return a fail low score.

            assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

            // Adjust best value for fail high cases
            if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
                bestValue = (bestValue * depth + beta) / (depth + 1);

            if (!moveCount)
                bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

            // If there is a move that produces search value greater than alpha,
            // we update the stats of searched moves.
            else if (bestMove)
            {
                update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched,
                                 depth, ttData.move);
                if (!PvNode)
                    ttMoveHistory << (bestMove == ttData.move ? 809 : -865);
            }

            // Bonus for prior quiet countermove that caused the fail low
            else if (!priorCapture && prevSq != SQ_NONE)
            {
                int bonusScale = -228;
                bonusScale -= (ss - 1)->statScore / 104;
                bonusScale += std::min(63 * depth, 508);
                bonusScale += 184 * ((ss - 1)->moveCount > 8);
                bonusScale += 143 * (!ss->inCheck && bestValue <= ss->staticEval - 92);
                bonusScale += 149 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 70);

                bonusScale = std::max(bonusScale, 0);

                const int scaledBonus = std::min(144 * depth - 92, 1365) * bonusScale;

                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                              scaledBonus * 400 / 32768);

                mainHistory[~us][((ss - 1)->currentMove).from_to()] << scaledBonus * 220 / 32768;

                if (type_of(pos.piece_on(prevSq)) != PAWN
                    && ((ss - 1)->currentMove).type_of() != PROMOTION)
                    pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
                      << scaledBonus * 1164 / 32768;
            }

            // Bonus for prior capture countermove that caused the fail low
            else if (priorCapture && prevSq != SQ_NONE)
            {
                Piece capturedPiece = pos.captured_piece();
                assert(capturedPiece != NO_PIECE);
                captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 964;
            }

            if (PvNode)
                bestValue = std::min(bestValue, maxValue);

            // If no good move is found and the previous position was ttPv, then the previous
            // opponent move is probably good and the new position is added to the search tree.
            if (bestValue <= alpha)
                ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

            // Write gathered information in transposition table. Note that the
            // static evaluation is saved as it was before correction history.
            if (!excludedMove && !(rootNode && pvIdx))
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                               bestValue >= beta    ? BOUND_LOWER
                               : PvNode && bestMove ? BOUND_EXACT
                                                    : BOUND_UPPER,
                               moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                               unadjustedStaticEval, tt.generation());

            // Adjust correction history  (restore master behavior exactly)
            if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
                && ((bestValue < ss->staticEval
                     && bestValue < beta)  // negative correction & no fail high
                    || (bestValue > ss->staticEval
                        && bestMove)))  // positive correction & no fail low
            {
                auto bonus = std::clamp(
                  int(bestValue - ss->staticEval) * depth / (8 + (bestValue > ss->staticEval)),
                  -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                update_correction_history(
                  pos, ss, *this, (1088 - 180 * (bestValue > ss->staticEval)) * bonus / 1024);
            }

            assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
        }

        return bestValue;
    }  // end of Search::Worker::search
}
// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    // Check if we have an upcoming move that draws by repetition
    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, bestMove;
    Value bestValue, value, futilityBase;
    bool  pvHit, givesCheck, capture;
    int   moveCount;

    // Step 1. Initialize node
    if (PvNode)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }

    bestMove    = Move::none();
    ss->inCheck = pos.checkers();
    moveCount   = 0;

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    // Step 2. Check for an immediate draw or maximum ply reached
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 3. Transposition table lookup
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    // Need further processing of the saved data
    ss->ttHit    = ttHit;
    ttData.move  = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    pvHit        = ttHit && ttData.is_pv;

    // At non-PV nodes we check for an early TT cutoff
    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    // Step 4. Static evaluation of the position
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);

        if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            // ttValue can be used as a better position evaluation
            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }

        // Stand pat. Return immediately if static value is at least beta
        if (bestValue >= beta)
        {
            if (!is_decisive(bestValue))
                bestValue = (bestValue + beta) / 2;
            if (!ss->ttHit)
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                               DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval,
                               tt.generation());
            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = ss->staticEval + 352;
    }

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory,
                                        (ss - 2)->continuationHistory};

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare to search
    // the moves. We presently use two stages of move generator in quiescence search:
    // captures, or evasions only when in check.
    MovePicker mp(pos, ttData.move, DEPTH_QS, &mainHistory, &lowPlyHistory, &captureHistory,
                  contHist, &pawnHistory, ss->ply);
    // --- QS Bayesian pre-gate v2 (off by default; cheap & hoisted) ---

    bool qsTbAvoid = false;
    if (g_bayesCfg.enabled && g_bayesCfg.tbGuard)
    {
        if (tbConfig.rootInTB)
            qsTbAvoid = true;
        else if (tbConfig.cardinality)
        {
            int _tbPiecesQS = pos.count<ALL_PIECES>();
            if (_tbPiecesQS <= tbConfig.cardinality)
                qsTbAvoid = true;
        }
    }

    const bool BayesQSOn =
      g_bayesCfg.enabled && g_bayesCfg.qsEnabled && !ss->inCheck && !PvNode && !qsTbAvoid;
    BayesConfig Bqs{};
    const bool  alphaDecisive  = is_decisive(alpha);
    int         qsSigmaCp      = 0;
    int         qsZQ8          = 0;
    int         qsZQ8Best      = 0;
    int         qsThrQ8        = 0;
    int         qsThrQ8Best    = 0;
    int         qsMuNodeBiasCp = 0;
    int         qsGated        = 0;
    if (BayesQSOn)
    {
        Bqs       = g_bayesCfg;
        qsSigmaCp = (Bqs.qsSigmaCp > 0 ? Bqs.qsSigmaCp : g_sigmaByDepth[0]);
        qsZQ8     = (Bqs.qsZStarQ8 > 0 ? Bqs.qsZStarQ8 : 360);
        qsZQ8Best = qsZQ8;
        if (Bqs.zTTBestBiasQ8 > 0)
            qsZQ8Best = std::max(0, qsZQ8 - Bqs.zTTBestBiasQ8);
        qsThrQ8     = mul_q8_cp_clamped(qsZQ8, std::max(1, qsSigmaCp));
        qsThrQ8Best = mul_q8_cp_clamped(qsZQ8Best, std::max(1, qsSigmaCp));
        if (Bqs.muTTAdjDiv > 0 && is_valid(ttData.value) && !is_decisive(ttData.value))
        {
            int adj = (int(ttData.value) - int(ss->staticEval)) / Bqs.muTTAdjDiv;
            if (Bqs.muTTAdjCapCp > 0)
            {
                if (adj > Bqs.muTTAdjCapCp)
                    adj = Bqs.muTTAdjCapCp;
                if (adj < -Bqs.muTTAdjCapCp)
                    adj = -Bqs.muTTAdjCapCp;
            }
            qsMuNodeBiasCp = adj;
        }
    }


    // Step 5. Loop through all pseudo-legal moves until no moves remain or a beta
    // cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {

        // Enforce hard cap on total verifications per node (not just gated ones)
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

        // Step 6. Pruning
        // Never prune evasions when in check; only prune in quiet (non-check) nodes
        if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
                && move.type_of() != PROMOTION)
            {
                if (moveCount > 2)
                    continue;

                Value futilityValue =
                  futilityBase
                  + (g_bayesCfg.enabled ? ValueOfCapturedPiece(CapturedPiecePre(pos, move))
                                        : PieceValue[pos.piece_on(move.to_sq())]);

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.
                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough
                // we can prune this move.
                if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::min(alpha, futilityBase);
                    continue;
                }
            }

            // Continuation history based pruning
            if (!capture
                && pawnHistory[pawn_history_index(pos)][pos.moved_piece(move)][move.to_sq()] < 7300)
                continue;

            // Do not search moves with bad enough SEE values
            if (!pos.see_ge(move, -78))
                continue;
        }

        //         // QS Bayesian pre-gate v2: history + TT µ-bias; hoisted thresholds
        // Reuse TT-driven Bayes bypass: skip QS Bayes gate if TT already indicates cutoff near beta
        const bool qsBypassBayes =
          BayesQSOn
          && (ttHit && is_valid(ttData.value) && !is_decisive(ttData.value)
              && ttData.depth >= (g_bayesCfg.qsTTLBMinDepth > 0 ? g_bayesCfg.qsTTLBMinDepth
                                                                : g_bayesCfg.ttLBMinDepth)
              && (((ttData.bound & BOUND_UPPER)
                   && (ttData.value + Value(g_bayesCfg.ttNearCp)) < beta)
                  || ((ttData.bound & BOUND_LOWER)
                      && (ttData.value - Value(g_bayesCfg.ttNearCp)) >= beta)));
        if (BayesQSOn && !qsBypassBayes && !alphaDecisive && capture
            && (Bqs.qsMaxGatedCaptures != 0 && qsGated < Bqs.qsMaxGatedCaptures))
        {
            Piece capPre = CapturedPiecePre(pos, move);
            if (move.type_of() == EN_PASSANT)
                capPre = make_piece(~pos.side_to_move(), PAWN);
            if (!(Bqs.qsMinCapturedValueCp > 0 && PieceValue[capPre] < Bqs.qsMinCapturedValueCp))
            {
                const Piece movedPre = pos.moved_piece(move);
                const int   qsDiv    = std::max(1, Bqs.histScaleDiv);
                int histTerm = captureHistory[movedPre][move.to_sq()][type_of(capPre)] / qsDiv;
                if (Bqs.histCapCp > 0)
                {
                    if (histTerm > Bqs.histCapCp)
                        histTerm = Bqs.histCapCp;
                    if (histTerm < -Bqs.histCapCp)
                        histTerm = -Bqs.histCapCp;
                }
                int promoDelta = 0;
                if (move.type_of() == PROMOTION)
                {
                    PieceType pt = move.promotion_type();
                    promoDelta   = PieceValue[make_piece(pos.side_to_move(), pt)]
                               - PieceValue[make_piece(pos.side_to_move(), PAWN)];
                    promoDelta = std::clamp(promoDelta, -2000, 2000);
                }
                int muCp = int(ss->staticEval) + (capPre == NO_PIECE ? 0 : g_capWeightVal[capPre])
                         + histTerm + Bqs.muBiasCp + qsMuNodeBiasCp + promoDelta;
                muCp      = std::clamp(muCp, -30000, 30000);
                int thrCp = int(alpha) + Bqs.qsAlphaMarginCp;
                // Keep within plausible centipawn range to avoid pathological sentinels
                thrCp           = std::clamp(thrCp, -30000, 30000);
                const int delta = thrCp - muCp;
                if (delta > 0)
                {
                    int rhsLocal = (ttData.move && move == ttData.move ? qsThrQ8Best : qsThrQ8);
                    if (move.type_of() == PROMOTION)
                        rhsLocal = std::max(0, rhsLocal - 80 * std::max(1, qsSigmaCp));
                    // Do the shift in 64-bit and clamp to avoid overflow
                    const int deltaQ8 =
                      int(std::clamp(1LL * delta * 256, -1LL * Q8_RAIL, 1LL * Q8_RAIL));
                    if (deltaQ8 >= rhsLocal)
                    {
                        ++qsGated;  // counted only when move is actually gated-out
                        continue;
                    }
                }
            }
        }

        // Step 7. Make and search the move
        do_move(pos, move, st, givesCheck, ss);

        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 8. Check for a new best move
        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

    // Step 9. Check for mate
    // All legal moves have been searched. A special case: if we are
    // in check and no legal moves were found, it is checkmate.
    if (ss->inCheck && bestValue == -VALUE_INFINITE)
    {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);  // Plies to mate from the root
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;


    Color us = pos.side_to_move();
    if (!ss->inCheck && !moveCount && !pos.non_pawn_material(us)
        && type_of(pos.captured_piece()) >= ROOK)
    {
        if (!((us == WHITE ? shift<NORTH>(pos.pieces(us, PAWN))
                           : shift<SOUTH>(pos.pieces(us, PAWN)))
              & ~pos.pieces()))  // no pawn pushes available
        {
            pos.state()->checkersBB = Rank1BB;  // search for legal king-moves only
            if (!MoveList<LEGAL>(pos).size())   // stalemate
                bestValue = VALUE_DRAW;
            pos.state()->checkersBB = 0;
        }
    }

    // Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.
    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

Depth Search::Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 757 / rootDelta + !i * reductionScale * 218 / 512 + 1200;
}

// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.
//
// elapsed_time() returns the actual time elapsed since the start of the search.
// This function is intended for use only when printing PV outputs, and not used
// for making decisions within the search algorithm itself.
TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }

Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);
}

namespace {
// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.
Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }


// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.
Value value_from_tt(Value v, int ply, int r50c) {

    if (!is_valid(v))
        return VALUE_NONE;

    // handle TB win or better
    if (is_win(v))
    {
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        return v - ply;
    }

    // handle TB loss or worse
    if (is_loss(v))
    {
        // Downgrade a potentially false mate score.
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        return v + ply;
    }

    return v;
}


// Adds current move and appends child pv[]
void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}


// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Search::Stack*  ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus = std::min(151 * depth - 91, 1730) + 302 * (bestMove == ttMove);
    int malus = std::min(951 * depth - 156, 2468) - 30 * quietsSearched.size();

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 957 / 1024);

        // Decrease stats for all non-best quiet moves
        for (Move move : quietsSearched)
            update_quiet_histories(pos, ss, workerThread, move, -malus);
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = type_of(g_bayesCfg.enabled ? CapturedPiecePre(pos, bestMove)
                                                   : pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 503 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched)
    {
        movedPiece = pos.moved_piece(move);
        capturedPiece =
          type_of(g_bayesCfg.enabled ? CapturedPiecePre(pos, move) : pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1157 / 1024;
    }
}


// Updates histories of the move pairs formed by moves
// at ply -1, -2, -3, -4, and -6 with current move.
void update_continuation_histories(Search::Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 1157}, {2, 648}, {3, 288}, {4, 576}, {5, 140}, {6, 441}}};

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << (bonus * weight / 1024) + 88 * (i < 2);
    }
}

// Updates move sorting heuristics

void update_quiet_histories(
  const Position& pos, Search::Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;  // Untuned to prevent duplicate effort

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.from_to()] << bonus * 761 / 1024;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * 955 / 1024);

    int pIndex = pawn_history_index(pos);
    workerThread.pawnHistory[pIndex][pos.moved_piece(move)][move.to_sq()]
      << (bonus * (bonus > 0 ? 800 : 500) / 1024) + 70;
}

}

// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'. Idea by Heinz van Saanen.
Move Skill::pick_best(const RootMoves& rootMoves, size_t multiPV) {
    static PRNG rng(now());  // PRNG sequence should be non-deterministic

    // RootMoves are already sorted by score in descending order
    Value  topScore = rootMoves[0].score;
    int    delta    = std::min(topScore - rootMoves[multiPV - 1].score, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms, both dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one is
    // random. Then we choose the move with the resulting highest score.
    for (size_t i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
        int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}


// Used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.
void SearchManager::check_time(Search::Worker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    if (ponder)
        return;

    if (
      // Later we rely on the fact that we can at least use the mainthread previous
      // root-search score and PV in a multithreaded environment to prove mated-in scores.
      worker.completedDepth >= 1
      && ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit))
          || (worker.limits.movetime && elapsed >= worker.limits.movetime)
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = worker.threads.abortedSearch = true;
}

// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
// Keeps the search based PV for as long as it is verified to maintain the game
// outcome, truncates afterwards. Finally, extends to mate the PV, providing a
// possible continuation (but not a proven mating line).
void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Position&                 pos,
                      RootMove&                 rootMove,
                      Value&                    v) {

    auto t_start      = std::chrono::steady_clock::now();
    int  moveOverhead = int(options["Move Overhead"]);
    bool rule50       = bool(options["Syzygy50MoveRule"]);

    // Do not use more than moveOverhead / 2 time, if time management is active
    auto time_abort = [&t_start, &moveOverhead, &limits]() -> bool {
        auto t_end = std::chrono::steady_clock::now();
        return limits.use_time_management()
            && 2 * std::chrono::duration<double, std::milli>(t_end - t_start).count()
                 > moveOverhead;
    };

    std::list<StateInfo> sts;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB.
    auto& stRoot = sts.emplace_back();
    pos.do_move(rootMove.pv[0], stRoot);
    int ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive score
    while (size_t(ply) < rootMove.pv.size())
    {
        Move& pvMove = rootMove.pv[ply];

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
            legalMoves.emplace_back(m);

        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves);
        RootMove&          rm     = *std::find(legalMoves.begin(), legalMoves.end(), pvMove);

        if (legalMoves[0].tbRank != rm.tbRank)
            break;

        ply++;

        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);

        // Do not allow for repetitions or drawing moves along the PV in TB regime
        if (config.rootInTB && ((rule50 && pos.is_draw(ply)) || pos.is_repetition(ply)))
        {
            pos.undo_move(pvMove);
            ply--;
            break;
        }

        // Full PV shown will thus be validated and end in TB.
        // If we cannot validate the full PV in time, we do not show it.
        if (config.rootInTB && time_abort())
            break;
    }

    // Resize the PV to the correct part
    rootMove.pv.resize(ply);

    // Step 2, now extend the PV to mate, as if the user explored syzygy-tables.info
    // using top ranked moves (minimal DTZ), which gives optimal mates only for simple
    // endgames e.g. KRvK.
    while (!(rule50 && pos.is_draw(0)))
    {
        if (time_abort())
            break;

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
        {
            auto&     rm = legalMoves.emplace_back(m);
            StateInfo tmpSI;
            pos.do_move(m, tmpSI);
            // Give a score of each move to break DTZ ties restricting opponent mobility,
            // but not giving the opponent a capture.
            for (const auto& mOpp : MoveList<LEGAL>(pos))
                rm.tbRank -= pos.capture(mOpp) ? 100 : 1;
            pos.undo_move(m);
        }

        // Mate found
        if (legalMoves.size() == 0)
            break;

        // Sort moves according to their above assigned rank.
        // This will break ties for moves with equal DTZ in rank_root_moves.
        std::stable_sort(
          legalMoves.begin(), legalMoves.end(),
          [](const Search::RootMove& a, const Search::RootMove& b) { return a.tbRank > b.tbRank; });

        // The winning side tries to minimize DTZ, the losing side maximizes it
        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves, true);

        // If DTZ is not available we might not find a mate, so we bail out
        if (!config.rootInTB || config.cardinality > 0)
            break;

        ply++;

        Move& pvMove = legalMoves[0].pv[0];
        rootMove.pv.push_back(pvMove);
        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);
    }

    // Finding a draw in this function is an exceptional case, that cannot happen when rule50 is false or
    // during engine game play, since we have a winning score, and play correctly
    // with TB support. However, it can be that a position is draw due to the 50 move
    // rule if it has been been reached on the board with a non-optimal 50 move counter
    // (e.g. 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter rounding
    // cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score can be
    // displayed if the engine did not find a drawing move yet, but eventually search
    // will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w - - 96 1 )
    if (pos.is_draw(0))
        v = VALUE_DRAW;

    // Undo the PV moves
    for (auto it = rootMove.pv.rbegin(); it != rootMove.pv.rend(); ++it)
        pos.undo_move(*it);

    // Inform if we couldn't get a full extension in time
    if (time_abort())
        sync_cout
          << "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
          << sync_endl;
}

void SearchManager::pv(Search::Worker&           worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) {

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    size_t     pvIdx     = worker.pvIdx;
    size_t     multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
    uint64_t   tbHits    = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);

    for (size_t i = 0; i < multiPV; ++i)
    {
        bool updated = rootMoves[i].score != -VALUE_INFINITE;

        if (depth == 1 && !updated && i > 0)
            continue;

        Depth d = updated ? depth : std::max(1, depth - 1);
        Value v = updated ? rootMoves[i].uciScore : rootMoves[i].previousScore;

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

        bool tb = worker.tbConfig.rootInTB && std::abs(v) <= VALUE_TB;
        v       = tb ? rootMoves[i].tbScore : v;

        bool isExact = i != pvIdx || tb || !updated;  // tablebase- and previous-scores are exact

        // Potentially correct and extend the PV, and in exceptional cases v
        if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
            && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);

        std::string pv;
        for (Move m : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";

        // Remove last whitespace
        if (!pv.empty())
            pv.pop_back();

        auto wdl   = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
        auto bound = rootMoves[i].scoreLowerbound
                     ? "lowerbound"
                     : (rootMoves[i].scoreUpperbound ? "upperbound" : "");

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
        info.score    = {v, pos};
        info.wdl      = wdl;

        if (!isExact)
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
        info.tbHits    = tbHits;
        info.pv        = pv;
        info.hashfull  = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI,
// otherwise in case of 'ponder on' we have nothing to think about.
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {

    StateInfo st;

    assert(pv.size() == 1);
    if (pv[0] == Move::none())
        return false;

    pos.do_move(pv[0], st, &tt);

    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
    if (ttHit)
    {
        if (MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}
}  // namespace Stockfish
