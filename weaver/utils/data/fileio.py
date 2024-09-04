import math
import awkward as ak
import tqdm
import re
import traceback
from .tools import _concat
from ..logger import _logger


def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop]
    return ak.Array(outputs)

import awkward as ak
import numpy as np
import numba as nb
@nb.njit
def retreive_length(target, builder):
    for t in target:
        for i, e in enumerate(list(t)[::-1]):
            if e!=0:
                t_end = len(t) - i
                break
        builder.integer(t_end)
    return builder
@nb.njit
def construct_jagged(target, length, builder):
    for t, l in zip(target, length):
        builder.begin_list()
        for i, e in enumerate(t[:l]):
            builder.real(e)
        builder.end_list()
    return builder

def _read_root(filepath, branches, load_range=None, treename=None):
    '''
    # for training with v4 and v5
    specific_vars = {
        # keywords: variable list
        "'BulkGravitonToHHTo4QTau' not in filepath and 'Spin0ToTT_VariableMass_W' not in filepath": {
            'label_H_ss': 0.,
            'label_H_leptauehadtau': 0.,
            'label_H_leptaumhadtau': 0.,
            'label_H_hadtauhadtau': 0.,
        },
        "'GluGluToBulkGravitonToHHTo4B' in filepath": {
            'label_H_ww2qsame': 0.,
            'label_H_ww2qsep': 0.,
            'label_H_ww3q_0c': 0.,
            'label_H_ww3q_1c': 0.,
            'label_H_ww3q_2c': 0.,
            'label_H_ww4q_0c': 0.,
            'label_H_ww4q_1c': 0.,
            'label_H_ww4q_2c': 0.,
            'label_H_wwevqq_0c': 0.,
            'label_H_wwevqq_1c': 0.,
            'label_H_wwhadtauvqq_0c': 0.,
            'label_H_wwhadtauvqq_1c': 0.,
            'label_H_wwleptauevqq_0c': 0.,
            'label_H_wwleptauevqq_1c': 0.,
            'label_H_wwleptaumvqq_0c': 0.,
            'label_H_wwleptaumvqq_1c': 0.,
            'label_H_wwmvqq_0c': 0.,
            'label_H_wwmvqq_1c': 0.,
            'label_Top_bc': 0.,
            'label_Top_bcq': 0.,
            'label_Top_bev': 0.,
            'label_Top_bhadtauv': 0.,
            'label_Top_bleptauev': 0.,
            'label_Top_bleptaumv': 0.,
            'label_Top_bmv': 0.,
            'label_Top_bq': 0.,
            'label_Top_bqq': 0.,
            'label_W_cq': 0.,
            'label_W_cq_b': 0.,
            'label_W_cq_c': 0.,
            'label_W_ev': 0.,
            'label_W_ev_b': 0.,
            'label_W_ev_c': 0.,
            'label_W_hadtauv': 0.,
            'label_W_hadtauv_b': 0.,
            'label_W_hadtauv_c': 0.,
            'label_W_leptauev': 0.,
            'label_W_leptauev_b': 0.,
            'label_W_leptauev_c': 0.,
            'label_W_leptaumv': 0.,
            'label_W_leptaumv_b': 0.,
            'label_W_leptaumv_c': 0.,
            'label_W_mv': 0.,
            'label_W_mv_b': 0.,
            'label_W_mv_c': 0.,
            'label_W_qq': 0.,
            'label_W_qq_b': 0.,
            'label_W_qq_c': 0.,
        },
        # "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' in filepath": {
        #     'label_H_wxwx4q_0c': 'label_H_ww4q_0c',
        #     'label_H_wxwx4q_1c': 'label_H_ww4q_1c',
        #     'label_H_wxwx4q_2c': 'label_H_ww4q_2c',
        #     'label_H_wxwx3q_0c': 'label_H_ww3q_0c',
        #     'label_H_wxwx3q_1c': 'label_H_ww3q_1c',
        #     'label_H_wxwx3q_2c': 'label_H_ww3q_2c',
        #     'label_H_wxwxevqq_0c': 'label_H_wwevqq_0c',
        #     'label_H_wxwxevqq_1c': 'label_H_wwevqq_1c',
        #     'label_H_wxwxmvqq_0c': 'label_H_wwmvqq_0c',
        #     'label_H_wxwxmvqq_1c': 'label_H_wwmvqq_1c',
        #     'label_H_wxwxleptauevqq_0c': 'label_H_wwleptauevqq_0c',
        #     'label_H_wxwxleptauevqq_1c': 'label_H_wwleptauevqq_1c',
        #     'label_H_wxwxleptaumvqq_0c': 'label_H_wwleptaumvqq_0c',
        #     'label_H_wxwxleptaumvqq_1c': 'label_H_wwleptaumvqq_1c',
        #     'label_H_wxwxhadtauvqq_0c': 'label_H_wwhadtauvqq_0c',
        #     'label_H_wxwxhadtauvqq_1c': 'label_H_wwhadtauvqq_1c',
        #     'label_H_w0w04q_0c': 0.,
        #     'label_H_w0w04q_1c': 0.,
        #     'label_H_w0w04q_2c': 0.,
        #     'label_H_w0w03q_0c': 0.,
        #     'label_H_w0w03q_1c': 0.,
        #     'label_H_w0w03q_2c': 0.,
        #     'label_H_w0w0evqq_0c': 0.,
        #     'label_H_w0w0evqq_1c': 0.,
        #     'label_H_w0w0mvqq_0c': 0.,
        #     'label_H_w0w0mvqq_1c': 0.,
        #     'label_H_w0w0leptauevqq_0c': 0.,
        #     'label_H_w0w0leptauevqq_1c': 0.,
        #     'label_H_w0w0leptaumvqq_0c': 0.,
        #     'label_H_w0w0leptaumvqq_1c': 0.,
        #     'label_H_w0w0hadtauvqq_0c': 0.,
        #     'label_H_w0w0hadtauvqq_1c': 0.,
        # },
        # "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' not in filepath": {
        #     'label_H_wxwx4q_0c': 0.,
        #     'label_H_wxwx4q_1c': 0.,
        #     'label_H_wxwx4q_2c': 0.,
        #     'label_H_wxwx3q_0c': 0.,
        #     'label_H_wxwx3q_1c': 0.,
        #     'label_H_wxwx3q_2c': 0.,
        #     'label_H_wxwxevqq_0c': 0.,
        #     'label_H_wxwxevqq_1c': 0.,
        #     'label_H_wxwxmvqq_0c': 0.,
        #     'label_H_wxwxmvqq_1c': 0.,
        #     'label_H_wxwxleptauevqq_0c': 0.,
        #     'label_H_wxwxleptauevqq_1c': 0.,
        #     'label_H_wxwxleptaumvqq_0c': 0.,
        #     'label_H_wxwxleptaumvqq_1c': 0.,
        #     'label_H_wxwxhadtauvqq_0c': 0.,
        #     'label_H_wxwxhadtauvqq_1c': 0.,
        #     'label_H_w0w04q_0c': 'label_H_ww4q_0c',
        #     'label_H_w0w04q_1c': 'label_H_ww4q_1c',
        #     'label_H_w0w04q_2c': 'label_H_ww4q_2c',
        #     'label_H_w0w03q_0c': 'label_H_ww3q_0c',
        #     'label_H_w0w03q_1c': 'label_H_ww3q_1c',
        #     'label_H_w0w03q_2c': 'label_H_ww3q_2c',
        #     'label_H_w0w0evqq_0c': 'label_H_wwevqq_0c',
        #     'label_H_w0w0evqq_1c': 'label_H_wwevqq_1c',
        #     'label_H_w0w0mvqq_0c': 'label_H_wwmvqq_0c',
        #     'label_H_w0w0mvqq_1c': 'label_H_wwmvqq_1c',
        #     'label_H_w0w0leptauevqq_0c': 'label_H_wwleptauevqq_0c',
        #     'label_H_w0w0leptauevqq_1c': 'label_H_wwleptauevqq_1c',
        #     'label_H_w0w0leptaumvqq_0c': 'label_H_wwleptaumvqq_0c',
        #     'label_H_w0w0leptaumvqq_1c': 'label_H_wwleptaumvqq_1c',
        #     'label_H_w0w0hadtauvqq_0c': 'label_H_wwhadtauvqq_0c',
        #     'label_H_w0w0hadtauvqq_1c': 'label_H_wwhadtauvqq_1c',
        # },
        "'QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8' in filepath": {
            'sample_isQCD_real': 1.,
        },
        "'QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8' not in filepath": {
            'sample_isQCD_real': 0.,
        },
    }
    specific_vars_included = {
        "True": ['label_H_ww4q_0c', 'label_H_ww4q_1c', 'label_H_ww4q_2c', 'label_H_ww3q_0c', 'label_H_ww3q_1c', 'label_H_ww3q_2c', 'label_H_wwevqq_0c', 'label_H_wwevqq_1c', 'label_H_wwmvqq_0c', 'label_H_wwmvqq_1c', 'label_H_wwleptauevqq_0c', 'label_H_wwleptauevqq_1c', 'label_H_wwleptaumvqq_0c', 'label_H_wwleptaumvqq_1c', 'label_H_wwhadtauvqq_0c', 'label_H_wwhadtauvqq_1c'],
    }
    '''
    '''
    # for training with v6
    specific_vars = {
        # keywords: variable list
        "'ChargedHiggs_HplusToBC_HminusToBC' not in filepath": {
            'label_H_bc': 0.,
        },
        "'QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8' in filepath": {
            'sample_isQCD_real': 1.,
        },
        "'QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8' not in filepath": {
            'sample_isQCD_real': 0.,
        },
        ### seprarate HWW and XWW
        # "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' in filepath": {
        #     'label_H_wxwx4q_0c': 'label_H_ww4q_0c', 'label_H_wxwx4q_1c': 'label_H_ww4q_1c', 'label_H_wxwx4q_2c': 'label_H_ww4q_2c','label_H_wxwx3q_0c': 'label_H_ww3q_0c', 'label_H_wxwx3q_1c': 'label_H_ww3q_1c', 'label_H_wxwx3q_2c': 'label_H_ww3q_2c', 'label_H_wxwxevqq_0c': 'label_H_wwevqq_0c', 'label_H_wxwxevqq_1c': 'label_H_wwevqq_1c', 'label_H_wxwxmvqq_0c': 'label_H_wwmvqq_0c', 'label_H_wxwxmvqq_1c': 'label_H_wwmvqq_1c', 'label_H_wxwxleptauevqq_0c': 'label_H_wwleptauevqq_0c', 'label_H_wxwxleptauevqq_1c': 'label_H_wwleptauevqq_1c', 'label_H_wxwxleptaumvqq_0c': 'label_H_wwleptaumvqq_0c', 'label_H_wxwxleptaumvqq_1c': 'label_H_wwleptaumvqq_1c', 'label_H_wxwxhadtauvqq_0c': 'label_H_wwhadtauvqq_0c', 'label_H_wxwxhadtauvqq_1c': 'label_H_wwhadtauvqq_1c', 
        #     'label_H_w0w04q_0c': 0., 'label_H_w0w04q_1c': 0., 'label_H_w0w04q_2c': 0., 'label_H_w0w03q_0c': 0., 'label_H_w0w03q_1c': 0., 'label_H_w0w03q_2c': 0., 'label_H_w0w0evqq_0c': 0., 'label_H_w0w0evqq_1c': 0., 'label_H_w0w0mvqq_0c': 0., 'label_H_w0w0mvqq_1c': 0., 'label_H_w0w0leptauevqq_0c': 0., 'label_H_w0w0leptauevqq_1c': 0., 'label_H_w0w0leptaumvqq_0c': 0., 'label_H_w0w0leptaumvqq_1c': 0., 'label_H_w0w0hadtauvqq_0c': 0., 'label_H_w0w0hadtauvqq_1c': 0.,
        # },
        # "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' not in filepath": {
        #     'label_H_wxwx4q_0c': 0., 'label_H_wxwx4q_1c': 0., 'label_H_wxwx4q_2c': 0., 'label_H_wxwx3q_0c': 0., 'label_H_wxwx3q_1c': 0., 'label_H_wxwx3q_2c': 0., 'label_H_wxwxevqq_0c': 0., 'label_H_wxwxevqq_1c': 0., 'label_H_wxwxmvqq_0c': 0., 'label_H_wxwxmvqq_1c': 0., 'label_H_wxwxleptauevqq_0c': 0., 'label_H_wxwxleptauevqq_1c': 0., 'label_H_wxwxleptaumvqq_0c': 0., 'label_H_wxwxleptaumvqq_1c': 0., 'label_H_wxwxhadtauvqq_0c': 0., 'label_H_wxwxhadtauvqq_1c': 0.,
        #     'label_H_w0w04q_0c': 'label_H_ww4q_0c', 'label_H_w0w04q_1c': 'label_H_ww4q_1c', 'label_H_w0w04q_2c': 'label_H_ww4q_2c', 'label_H_w0w03q_0c': 'label_H_ww3q_0c', 'label_H_w0w03q_1c': 'label_H_ww3q_1c', 'label_H_w0w03q_2c': 'label_H_ww3q_2c', 'label_H_w0w0evqq_0c': 'label_H_wwevqq_0c', 'label_H_w0w0evqq_1c': 'label_H_wwevqq_1c', 'label_H_w0w0mvqq_0c': 'label_H_wwmvqq_0c', 'label_H_w0w0mvqq_1c': 'label_H_wwmvqq_1c', 'label_H_w0w0leptauevqq_0c': 'label_H_wwleptauevqq_0c', 'label_H_w0w0leptauevqq_1c': 'label_H_wwleptauevqq_1c', 'label_H_w0w0leptaumvqq_0c': 'label_H_wwleptaumvqq_0c', 'label_H_w0w0leptaumvqq_1c': 'label_H_wwleptaumvqq_1c', 'label_H_w0w0hadtauvqq_0c': 'label_H_wwhadtauvqq_0c', 'label_H_w0w0hadtauvqq_1c': 'label_H_wwhadtauvqq_1c',
        # },
        ### new: separate HWW, XWW and XWW*
        "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' in filepath": {
            'alpha_dau_bal': 'abs(fj_gendau1_mass - fj_gendau2_mass) / (fj_gendau1_mass + fj_gendau2_mass)',
            'label_H_wxwxunbal4q_0c': '(alpha_dau_bal >= 0.1) & (label_H_ww4q_0c > 0)', 'label_H_wxwxunbal4q_1c': '(alpha_dau_bal >= 0.1) & (label_H_ww4q_1c > 0)', 'label_H_wxwxunbal4q_2c': '(alpha_dau_bal >= 0.1) & (label_H_ww4q_2c > 0)', 'label_H_wxwxunbal3q_0c': '(alpha_dau_bal >= 0.1) & (label_H_ww3q_0c > 0)', 'label_H_wxwxunbal3q_1c': '(alpha_dau_bal >= 0.1) & (label_H_ww3q_1c > 0)', 'label_H_wxwxunbal3q_2c': '(alpha_dau_bal >= 0.1) & (label_H_ww3q_2c > 0)', 'label_H_wxwxunbalevqq_0c': '(alpha_dau_bal >= 0.1) & (label_H_wwevqq_0c > 0)', 'label_H_wxwxunbalevqq_1c': '(alpha_dau_bal >= 0.1) & (label_H_wwevqq_1c > 0)', 'label_H_wxwxunbalmvqq_0c': '(alpha_dau_bal >= 0.1) & (label_H_wwmvqq_0c > 0)', 'label_H_wxwxunbalmvqq_1c': '(alpha_dau_bal >= 0.1) & (label_H_wwmvqq_1c > 0)', 'label_H_wxwxunballeptauevqq_0c': '(alpha_dau_bal >= 0.1) & (label_H_wwleptauevqq_0c > 0)', 'label_H_wxwxunballeptauevqq_1c': '(alpha_dau_bal >= 0.1) & (label_H_wwleptauevqq_1c > 0)', 'label_H_wxwxunballeptaumvqq_0c': '(alpha_dau_bal >= 0.1) & (label_H_wwleptaumvqq_0c > 0)', 'label_H_wxwxunballeptaumvqq_1c': '(alpha_dau_bal >= 0.1) & (label_H_wwleptaumvqq_1c > 0)', 'label_H_wxwxunbalhadtauvqq_0c': '(alpha_dau_bal >= 0.1) & (label_H_wwhadtauvqq_0c > 0)', 'label_H_wxwxunbalhadtauvqq_1c': '(alpha_dau_bal >= 0.1) & (label_H_wwhadtauvqq_1c > 0)', 
            'label_H_wxwxbal4q_0c': '(alpha_dau_bal < 0.1) & (label_H_ww4q_0c > 0)', 'label_H_wxwxbal4q_1c': '(alpha_dau_bal < 0.1) & (label_H_ww4q_1c > 0)', 'label_H_wxwxbal4q_2c': '(alpha_dau_bal < 0.1) & (label_H_ww4q_2c > 0)', 'label_H_wxwxbal3q_0c': '(alpha_dau_bal < 0.1) & (label_H_ww3q_0c > 0)', 'label_H_wxwxbal3q_1c': '(alpha_dau_bal < 0.1) & (label_H_ww3q_1c > 0)', 'label_H_wxwxbal3q_2c': '(alpha_dau_bal < 0.1) & (label_H_ww3q_2c > 0)', 'label_H_wxwxbalevqq_0c': '(alpha_dau_bal < 0.1) & (label_H_wwevqq_0c > 0)', 'label_H_wxwxbalevqq_1c': '(alpha_dau_bal < 0.1) & (label_H_wwevqq_1c > 0)', 'label_H_wxwxbalmvqq_0c': '(alpha_dau_bal < 0.1) & (label_H_wwmvqq_0c > 0)', 'label_H_wxwxbalmvqq_1c': '(alpha_dau_bal < 0.1) & (label_H_wwmvqq_1c > 0)', 'label_H_wxwxballeptauevqq_0c': '(alpha_dau_bal < 0.1) & (label_H_wwleptauevqq_0c > 0)', 'label_H_wxwxballeptauevqq_1c': '(alpha_dau_bal < 0.1) & (label_H_wwleptauevqq_1c > 0)', 'label_H_wxwxballeptaumvqq_0c': '(alpha_dau_bal < 0.1) & (label_H_wwleptaumvqq_0c > 0)', 'label_H_wxwxballeptaumvqq_1c': '(alpha_dau_bal < 0.1) & (label_H_wwleptaumvqq_1c > 0)', 'label_H_wxwxbalhadtauvqq_0c': '(alpha_dau_bal < 0.1) & (label_H_wwhadtauvqq_0c > 0)', 'label_H_wxwxbalhadtauvqq_1c': '(alpha_dau_bal < 0.1) & (label_H_wwhadtauvqq_1c > 0)', 
            'label_H_w0w04q_0c': 0., 'label_H_w0w04q_1c': 0., 'label_H_w0w04q_2c': 0., 'label_H_w0w03q_0c': 0., 'label_H_w0w03q_1c': 0., 'label_H_w0w03q_2c': 0., 'label_H_w0w0evqq_0c': 0., 'label_H_w0w0evqq_1c': 0., 'label_H_w0w0mvqq_0c': 0., 'label_H_w0w0mvqq_1c': 0., 'label_H_w0w0leptauevqq_0c': 0., 'label_H_w0w0leptauevqq_1c': 0., 'label_H_w0w0leptaumvqq_0c': 0., 'label_H_w0w0leptaumvqq_1c': 0., 'label_H_w0w0hadtauvqq_0c': 0., 'label_H_w0w0hadtauvqq_1c': 0.,
        },
        "'BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh' not in filepath": {
            'alpha_dau_bal': -99.,
            'label_H_wxwxunbal4q_0c': 0., 'label_H_wxwxunbal4q_1c': 0., 'label_H_wxwxunbal4q_2c': 0., 'label_H_wxwxunbal3q_0c': 0., 'label_H_wxwxunbal3q_1c': 0., 'label_H_wxwxunbal3q_2c': 0., 'label_H_wxwxunbalevqq_0c': 0., 'label_H_wxwxunbalevqq_1c': 0., 'label_H_wxwxunbalmvqq_0c': 0., 'label_H_wxwxunbalmvqq_1c': 0., 'label_H_wxwxunballeptauevqq_0c': 0., 'label_H_wxwxunballeptauevqq_1c': 0., 'label_H_wxwxunballeptaumvqq_0c': 0., 'label_H_wxwxunballeptaumvqq_1c': 0., 'label_H_wxwxunbalhadtauvqq_0c': 0., 'label_H_wxwxunbalhadtauvqq_1c': 0.,
            'label_H_wxwxbal4q_0c': 0., 'label_H_wxwxbal4q_1c': 0., 'label_H_wxwxbal4q_2c': 0., 'label_H_wxwxbal3q_0c': 0., 'label_H_wxwxbal3q_1c': 0., 'label_H_wxwxbal3q_2c': 0., 'label_H_wxwxbalevqq_0c': 0., 'label_H_wxwxbalevqq_1c': 0., 'label_H_wxwxbalmvqq_0c': 0., 'label_H_wxwxbalmvqq_1c': 0., 'label_H_wxwxballeptauevqq_0c': 0., 'label_H_wxwxballeptauevqq_1c': 0., 'label_H_wxwxballeptaumvqq_0c': 0., 'label_H_wxwxballeptaumvqq_1c': 0., 'label_H_wxwxbalhadtauvqq_0c': 0., 'label_H_wxwxbalhadtauvqq_1c': 0.,
            'label_H_w0w04q_0c': 'label_H_ww4q_0c', 'label_H_w0w04q_1c': 'label_H_ww4q_1c', 'label_H_w0w04q_2c': 'label_H_ww4q_2c', 'label_H_w0w03q_0c': 'label_H_ww3q_0c', 'label_H_w0w03q_1c': 'label_H_ww3q_1c', 'label_H_w0w03q_2c': 'label_H_ww3q_2c', 'label_H_w0w0evqq_0c': 'label_H_wwevqq_0c', 'label_H_w0w0evqq_1c': 'label_H_wwevqq_1c', 'label_H_w0w0mvqq_0c': 'label_H_wwmvqq_0c', 'label_H_w0w0mvqq_1c': 'label_H_wwmvqq_1c', 'label_H_w0w0leptauevqq_0c': 'label_H_wwleptauevqq_0c', 'label_H_w0w0leptauevqq_1c': 'label_H_wwleptauevqq_1c', 'label_H_w0w0leptaumvqq_0c': 'label_H_wwleptaumvqq_0c', 'label_H_w0w0leptaumvqq_1c': 'label_H_wwleptaumvqq_1c', 'label_H_w0w0hadtauvqq_0c': 'label_H_wwhadtauvqq_0c', 'label_H_w0w0hadtauvqq_1c': 'label_H_wwhadtauvqq_1c',
        },
    }
    specific_vars_included = {
        # "True": ['label_H_ww4q_0c', 'label_H_ww4q_1c', 'label_H_ww4q_2c', 'label_H_ww3q_0c', 'label_H_ww3q_1c', 'label_H_ww3q_2c', 'label_H_wwevqq_0c', 'label_H_wwevqq_1c', 'label_H_wwmvqq_0c', 'label_H_wwmvqq_1c', 'label_H_wwleptauevqq_0c', 'label_H_wwleptauevqq_1c', 'label_H_wwleptaumvqq_0c', 'label_H_wwleptaumvqq_1c', 'label_H_wwhadtauvqq_0c', 'label_H_wwhadtauvqq_1c'],
        "True": ['label_H_ww4q_0c', 'label_H_ww4q_1c', 'label_H_ww4q_2c', 'label_H_ww3q_0c', 'label_H_ww3q_1c', 'label_H_ww3q_2c', 'label_H_wwevqq_0c', 'label_H_wwevqq_1c', 'label_H_wwmvqq_0c', 'label_H_wwmvqq_1c', 'label_H_wwleptauevqq_0c', 'label_H_wwleptauevqq_1c', 'label_H_wwleptaumvqq_0c', 'label_H_wwleptaumvqq_1c', 'label_H_wwhadtauvqq_0c', 'label_H_wwhadtauvqq_1c', 'fj_gendau1_mass', 'fj_gendau2_mass'],
    }
    '''
    # for training with v7 and on
    specific_vars = {}
    specific_vars_included = {}
    # configuration for pheno dataset
    specific_vars = {
        "filepath.endswith('40ifb_part0.root') or re.search(r'mixed_ntuple/ntuples_\d*[1267].root$', filepath)": {'file_no': 0},
        "filepath.endswith('40ifb_part1.root') or re.search(r'mixed_ntuple/ntuples_\d*[3489].root$', filepath)": {'file_no': 1},
        "filepath.endswith('20ifb.root') or re.search(r'mixed_ntuple/ntuples_\d*[05].root$', filepath)": {'file_no': 2},
    }

    def remove_branch(branches, filepath):
        for expr, new_branch_dict in specific_vars.items():
            if eval(expr):
                branches = list(set(branches) - set(new_branch_dict.keys()))
        for expr, new_branch_list in specific_vars_included.items():
            if eval(expr):
                branches = list(set(branches) | set(new_branch_list))
        return branches

    def add_new_branch(outputs, filepath, nent):
        for expr, new_branch_dict in specific_vars.items():
            if eval(expr):
                for b, v in new_branch_dict.items():
                    if isinstance(v, (int, float)):
                        outputs[b] = np.zeros(nent, dtype=np.float32) + v
                    elif isinstance(v, str):
                        outputs[b] = ak.values_astype(ak.numexpr.evaluate(v, outputs), np.float32)

    _branches = branches.copy()
    _branches = remove_branch(_branches, filepath)

    import uproot
    with uproot.open(filepath, object_cache=None, array_cache=None) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    'Need to specify `treename` as more than one trees are found in file %s: %s' %
                    (filepath, str(branches)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        outputs = tree.arrays(filter_name=_branches, entry_start=start, entry_stop=stop)

    nent = len(outputs)
    add_new_branch(outputs, filepath, nent)
    # print('//', filepath, ak.fields(outputs))

    return outputs


def _read_awkd(filepath, branches, load_range=None):
    import awkward0
    with awkward0.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = ak.from_awkward0(v[start:stop])
    return ak.Array(outputs)


def _read_parquet(filepath, branches, load_range=None):
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return outputs


def _read_files(filelist, branches, load_ranges=None, show_progressbar=False, **kwargs):
    import os
    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for i_file, filepath in enumerate(filelist):
        ext = os.path.splitext(filepath)[1]
        load_range = None if load_ranges is None else load_ranges[i_file]
        if ext not in ('.h5', '.root', '.awkd', '.parquet'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range, treename=kwargs.get('treename', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
            elif ext == '.parquet':
                a = _read_parquet(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if a is not None:
            table.append(a)
    table = _concat(table)  # ak.Array
    if len(table) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        tree = fout.mktree(treename, {k: v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            tree.extend({k: v[start:start + step] for k, v in table.items()})
            start += step
