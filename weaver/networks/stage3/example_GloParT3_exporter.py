import os
import ast
import math
import torch
import torch.nn as nn
from torch import Tensor
import tqdm
import time
from collections import defaultdict, Counter
import numpy as np

from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformerTagger_ncoll


class GlobalParticleTransformerExporter(ParticleTransformerTagger_ncoll):
    def __init__(self, *args, **kwargs) -> None:

        assert kwargs['for_inference'], "for_inference must be set to True for exporter"

        self.input_highlevel_dim = kwargs.pop('input_highlevel_dim', 0)
        fc_params = kwargs.pop('fc_params')
        num_nodes = kwargs.pop('num_nodes')
        self.num_nodes_cls = kwargs.pop('num_nodes_cls')
        aux_num_nodes = kwargs.pop('aux_num_nodes')
        aux_fc_params = kwargs.pop('aux_fc_params')
        self.glopart_version = kwargs.pop('version')
        self.num_output_nodes = kwargs.pop('num_output_nodes')

        kwargs['fc_params'] = None # reset fc_params to None so that the ParT model will just output the last embed layer
        super().__init__(*args, **kwargs)

        # fc:
        fcs = []
        in_dim = kwargs['embed_dims'][-1]
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, num_nodes))
        self.fc = nn.Sequential(*fcs)

        # aux_fc:
        fcs = []
        in_dim = kwargs['embed_dims'][-1] + self.input_highlevel_dim # concat high-level input dims to the embed layer
        for out_dim, drop_rate in aux_fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, aux_num_nodes))
        self.aux_fc = nn.Sequential(*fcs)
    
    def postprocess(self, x, output):

        if self.glopart_version == 'beta1':
            # 313+313 output values

            class_labels = ['label_Top_bWcs', 'label_Top_bWqq', 'label_Top_bWc', 'label_Top_bWs', 'label_Top_bWq', 'label_Top_bWev', 'label_Top_bWmv', 'label_Top_bWtauev', 'label_Top_bWtaumv', 'label_Top_bWtauhv', 'label_Top_Wcs', 'label_Top_Wqq', 'label_Top_Wev', 'label_Top_Wmv', 'label_Top_Wtauev', 'label_Top_Wtaumv', 'label_Top_Wtauhv', 'label_H_bb', 'label_H_cc', 'label_H_ss', 'label_H_qq', 'label_H_bc', 'label_H_cs', 'label_H_gg', 'label_H_ee', 'label_H_mm', 'label_H_tauhtaue', 'label_H_tauhtaum', 'label_H_tauhtauh', 'label_H_WW_cscs', 'label_H_WW_csqq', 'label_H_WW_qqqq', 'label_H_WW_csc', 'label_H_WW_css', 'label_H_WW_csq', 'label_H_WW_qqc', 'label_H_WW_qqs', 'label_H_WW_qqq', 'label_H_WW_csev', 'label_H_WW_qqev', 'label_H_WW_csmv', 'label_H_WW_qqmv', 'label_H_WW_cstauev', 'label_H_WW_qqtauev', 'label_H_WW_cstaumv', 'label_H_WW_qqtaumv', 'label_H_WW_cstauhv', 'label_H_WW_qqtauhv', 'label_H_WxWx_cscs', 'label_H_WxWx_csqq', 'label_H_WxWx_qqqq', 'label_H_WxWx_csc', 'label_H_WxWx_css', 'label_H_WxWx_csq', 'label_H_WxWx_qqc', 'label_H_WxWx_qqs', 'label_H_WxWx_qqq', 'label_H_WxWx_csev', 'label_H_WxWx_qqev', 'label_H_WxWx_csmv', 'label_H_WxWx_qqmv', 'label_H_WxWx_cstauev', 'label_H_WxWx_qqtauev', 'label_H_WxWx_cstaumv', 'label_H_WxWx_qqtaumv', 'label_H_WxWx_cstauhv', 'label_H_WxWx_qqtauhv', 'label_H_WxWxStar_cscs', 'label_H_WxWxStar_csqq', 'label_H_WxWxStar_qqqq', 'label_H_WxWxStar_csc', 'label_H_WxWxStar_css', 'label_H_WxWxStar_csq', 'label_H_WxWxStar_qqc', 'label_H_WxWxStar_qqs', 'label_H_WxWxStar_qqq', 'label_H_WxWxStar_csev', 'label_H_WxWxStar_qqev', 'label_H_WxWxStar_csmv', 'label_H_WxWxStar_qqmv', 'label_H_WxWxStar_cstauev', 'label_H_WxWxStar_qqtauev', 'label_H_WxWxStar_cstaumv', 'label_H_WxWxStar_qqtaumv', 'label_H_WxWxStar_cstauhv', 'label_H_WxWxStar_qqtauhv', 'label_H_ZZ_bbbb', 'label_H_ZZ_bbcc', 'label_H_ZZ_bbss', 'label_H_ZZ_bbqq', 'label_H_ZZ_cccc', 'label_H_ZZ_ccss', 'label_H_ZZ_ccqq', 'label_H_ZZ_ssss', 'label_H_ZZ_ssqq', 'label_H_ZZ_qqqq', 'label_H_ZZ_bbb', 'label_H_ZZ_bbc', 'label_H_ZZ_bbs', 'label_H_ZZ_bbq', 'label_H_ZZ_ccb', 'label_H_ZZ_ccc', 'label_H_ZZ_ccs', 'label_H_ZZ_ccq', 'label_H_ZZ_ssb', 'label_H_ZZ_ssc', 'label_H_ZZ_sss', 'label_H_ZZ_ssq', 'label_H_ZZ_qqb', 'label_H_ZZ_qqc', 'label_H_ZZ_qqs', 'label_H_ZZ_qqq', 'label_H_ZZ_bbee', 'label_H_ZZ_bbmm', 'label_H_ZZ_bbe', 'label_H_ZZ_bbm', 'label_H_ZZ_bee', 'label_H_ZZ_bmm', 'label_H_ZZ_bbtauhtaue', 'label_H_ZZ_bbtauhtaum', 'label_H_ZZ_bbtauhtauh', 'label_H_ZZ_btauhtaue', 'label_H_ZZ_btauhtaum', 'label_H_ZZ_btauhtauh', 'label_H_ZZ_ccee', 'label_H_ZZ_ccmm', 'label_H_ZZ_cce', 'label_H_ZZ_ccm', 'label_H_ZZ_cee', 'label_H_ZZ_cmm', 'label_H_ZZ_cctauhtaue', 'label_H_ZZ_cctauhtaum', 'label_H_ZZ_cctauhtauh', 'label_H_ZZ_ctauhtaue', 'label_H_ZZ_ctauhtaum', 'label_H_ZZ_ctauhtauh', 'label_H_ZZ_ssee', 'label_H_ZZ_ssmm', 'label_H_ZZ_sse', 'label_H_ZZ_ssm', 'label_H_ZZ_see', 'label_H_ZZ_smm', 'label_H_ZZ_sstauhtaue', 'label_H_ZZ_sstauhtaum', 'label_H_ZZ_sstauhtauh', 'label_H_ZZ_stauhtaue', 'label_H_ZZ_stauhtaum', 'label_H_ZZ_stauhtauh', 'label_H_ZZ_qqee', 'label_H_ZZ_qqmm', 'label_H_ZZ_qqe', 'label_H_ZZ_qqm', 'label_H_ZZ_qee', 'label_H_ZZ_qmm', 'label_H_ZZ_qqtauhtaue', 'label_H_ZZ_qqtauhtaum', 'label_H_ZZ_qqtauhtauh', 'label_H_ZZ_qtauhtaue', 'label_H_ZZ_qtauhtaum', 'label_H_ZZ_qtauhtauh', 'label_H_ZxZx_bbbb', 'label_H_ZxZx_bbcc', 'label_H_ZxZx_bbss', 'label_H_ZxZx_bbqq', 'label_H_ZxZx_cccc', 'label_H_ZxZx_ccss', 'label_H_ZxZx_ccqq', 'label_H_ZxZx_ssss', 'label_H_ZxZx_ssqq', 'label_H_ZxZx_qqqq', 'label_H_ZxZx_bbb', 'label_H_ZxZx_bbc', 'label_H_ZxZx_bbs', 'label_H_ZxZx_bbq', 'label_H_ZxZx_ccb', 'label_H_ZxZx_ccc', 'label_H_ZxZx_ccs', 'label_H_ZxZx_ccq', 'label_H_ZxZx_ssb', 'label_H_ZxZx_ssc', 'label_H_ZxZx_sss', 'label_H_ZxZx_ssq', 'label_H_ZxZx_qqb', 'label_H_ZxZx_qqc', 'label_H_ZxZx_qqs', 'label_H_ZxZx_qqq', 'label_H_ZxZx_bbee', 'label_H_ZxZx_bbmm', 'label_H_ZxZx_bbe', 'label_H_ZxZx_bbm', 'label_H_ZxZx_bee', 'label_H_ZxZx_bmm', 'label_H_ZxZx_bbtauhtaue', 'label_H_ZxZx_bbtauhtaum', 'label_H_ZxZx_bbtauhtauh', 'label_H_ZxZx_btauhtaue', 'label_H_ZxZx_btauhtaum', 'label_H_ZxZx_btauhtauh', 'label_H_ZxZx_ccee', 'label_H_ZxZx_ccmm', 'label_H_ZxZx_cce', 'label_H_ZxZx_ccm', 'label_H_ZxZx_cee', 'label_H_ZxZx_cmm', 'label_H_ZxZx_cctauhtaue', 'label_H_ZxZx_cctauhtaum', 'label_H_ZxZx_cctauhtauh', 'label_H_ZxZx_ctauhtaue', 'label_H_ZxZx_ctauhtaum', 'label_H_ZxZx_ctauhtauh', 'label_H_ZxZx_ssee', 'label_H_ZxZx_ssmm', 'label_H_ZxZx_sse', 'label_H_ZxZx_ssm', 'label_H_ZxZx_see', 'label_H_ZxZx_smm', 'label_H_ZxZx_sstauhtaue', 'label_H_ZxZx_sstauhtaum', 'label_H_ZxZx_sstauhtauh', 'label_H_ZxZx_stauhtaue', 'label_H_ZxZx_stauhtaum', 'label_H_ZxZx_stauhtauh', 'label_H_ZxZx_qqee', 'label_H_ZxZx_qqmm', 'label_H_ZxZx_qqe', 'label_H_ZxZx_qqm', 'label_H_ZxZx_qee', 'label_H_ZxZx_qmm', 'label_H_ZxZx_qqtauhtaue', 'label_H_ZxZx_qqtauhtaum', 'label_H_ZxZx_qqtauhtauh', 'label_H_ZxZx_qtauhtaue', 'label_H_ZxZx_qtauhtaum', 'label_H_ZxZx_qtauhtauh', 'label_H_ZxZxStar_bbbb', 'label_H_ZxZxStar_bbcc', 'label_H_ZxZxStar_bbss', 'label_H_ZxZxStar_bbqq', 'label_H_ZxZxStar_cccc', 'label_H_ZxZxStar_ccss', 'label_H_ZxZxStar_ccqq', 'label_H_ZxZxStar_ssss', 'label_H_ZxZxStar_ssqq', 'label_H_ZxZxStar_qqqq', 'label_H_ZxZxStar_bbb', 'label_H_ZxZxStar_bbc', 'label_H_ZxZxStar_bbs', 'label_H_ZxZxStar_bbq', 'label_H_ZxZxStar_ccb', 'label_H_ZxZxStar_ccc', 'label_H_ZxZxStar_ccs', 'label_H_ZxZxStar_ccq', 'label_H_ZxZxStar_ssb', 'label_H_ZxZxStar_ssc', 'label_H_ZxZxStar_sss', 'label_H_ZxZxStar_ssq', 'label_H_ZxZxStar_qqb', 'label_H_ZxZxStar_qqc', 'label_H_ZxZxStar_qqs', 'label_H_ZxZxStar_qqq', 'label_H_ZxZxStar_bbee', 'label_H_ZxZxStar_bbmm', 'label_H_ZxZxStar_bbe', 'label_H_ZxZxStar_bbm', 'label_H_ZxZxStar_bee', 'label_H_ZxZxStar_bmm', 'label_H_ZxZxStar_bbtauhtaue', 'label_H_ZxZxStar_bbtauhtaum', 'label_H_ZxZxStar_bbtauhtauh', 'label_H_ZxZxStar_btauhtaue', 'label_H_ZxZxStar_btauhtaum', 'label_H_ZxZxStar_btauhtauh', 'label_H_ZxZxStar_ccee', 'label_H_ZxZxStar_ccmm', 'label_H_ZxZxStar_cce', 'label_H_ZxZxStar_ccm', 'label_H_ZxZxStar_cee', 'label_H_ZxZxStar_cmm', 'label_H_ZxZxStar_cctauhtaue', 'label_H_ZxZxStar_cctauhtaum', 'label_H_ZxZxStar_cctauhtauh', 'label_H_ZxZxStar_ctauhtaue', 'label_H_ZxZxStar_ctauhtaum', 'label_H_ZxZxStar_ctauhtauh', 'label_H_ZxZxStar_ssee', 'label_H_ZxZxStar_ssmm', 'label_H_ZxZxStar_sse', 'label_H_ZxZxStar_ssm', 'label_H_ZxZxStar_see', 'label_H_ZxZxStar_smm', 'label_H_ZxZxStar_sstauhtaue', 'label_H_ZxZxStar_sstauhtaum', 'label_H_ZxZxStar_sstauhtauh', 'label_H_ZxZxStar_stauhtaue', 'label_H_ZxZxStar_stauhtaum', 'label_H_ZxZxStar_stauhtauh', 'label_H_ZxZxStar_qqee', 'label_H_ZxZxStar_qqmm', 'label_H_ZxZxStar_qqe', 'label_H_ZxZxStar_qqm', 'label_H_ZxZxStar_qee', 'label_H_ZxZxStar_qmm', 'label_H_ZxZxStar_qqtauhtaue', 'label_H_ZxZxStar_qqtauhtaum', 'label_H_ZxZxStar_qqtauhtauh', 'label_H_ZxZxStar_qtauhtaue', 'label_H_ZxZxStar_qtauhtaum', 'label_H_ZxZxStar_qtauhtauh', 'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others']
            aux_class_labels = ['label_Top_bWcs', 'label_Top_bWqq', 'label_Top_bWc', 'label_Top_bWs', 'label_Top_bWq', 'label_Top_bWev', 'label_Top_bWmv', 'label_Top_bWtauev', 'label_Top_bWtaumv', 'label_Top_bWtauhv', 'label_Top_Wcs', 'label_Top_Wqq', 'label_Top_Wev', 'label_Top_Wmv', 'label_Top_Wtauev', 'label_Top_Wtaumv', 'label_Top_Wtauhv', 'label_W_cs', 'label_W_qq', 'label_Z_bb', 'label_Z_cc', 'label_Z_ss', 'label_Z_qq', 'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others']

            # assign output variable name (313+313+28 outputs)
            output_varname = [l.replace('_', '').replace('label', 'prob') for l in class_labels]
            output_varname += [l.replace('_', '').replace('label', 'massCorr') for l in class_labels]
            output_varname += [l.replace('_', '').replace('label', 'probWithMass') for l in aux_class_labels]

            assert len(output_varname) == 313+313+28 # beta1 setup

            # define variable replacement map
            replace_map = {name: f'output[:,{i}]' for i, name in enumerate(output_varname)}

            def get_expr(expression):
                tree = ast.parse(expression, mode='eval')
                # variables = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                for n in ast.walk(tree):
                    if isinstance(n, ast.Name) and n.id in replace_map:
                        n.id = replace_map[n.id]
                return ast.unparse(tree)

            output_nodes_expr = [
                ## ===== classification nodes ===== #
                'probHbb', # Xbb
                'probHcc', # Xcc
                'probHcs', # Xcs
                'probHss + probHqq*2', # Xqq
                'probHtauhtaue', # Xtauhtaue
                'probHtauhtaum', # Xtauhtaum
                'probHtauhtauh', # Xtauhtauh
                'probHWWcscs + probHWWcsqq + probHWWqqqq', # XWW4q
                'probHWWcsc + probHWWcss + probHWWcsq + probHWWqqc + probHWWqqs + probHWWqqq', # XWW3q
                'probHWWcsev + probHWWqqev', # XWWqqev
                'probHWWcsmv + probHWWqqmv', # XWWqqmv
                'probTopbWcs + probTopbWqq', # TopbWqq
                'probTopbWc + probTopbWs + probTopbWq', # TopbWq
                'probTopbWev', # TopbWev
                'probTopbWmv', # TopbWmv
                'probTopbWtauhv', # TopbWtauhv
                'probQCDbb + probQCDcc + probQCDb + probQCDc + probQCDothers', # QCD
                ## ===== regression nodes ===== ##
                '(massCorrHbb*probHbb + massCorrHcc*probHcc + massCorrHss*probHss + massCorrHqq*probHqq + massCorrHbc*probHbc + massCorrHcs*probHcs) / (probHbb + probHcc + probHss + probHqq + probHbc + probHcs)', # massCorrX2p
                '(massCorrHWWcscs*probHWWcscs + massCorrHWWcsqq*probHWWcsqq + massCorrHWWqqqq*probHWWqqqq) / (probHWWcscs + probHWWcsqq + probHWWqqqq)', # massCorrXWW (4q)
                '(massCorrTopbWcs*probTopbWcs + massCorrTopbWqq*probTopbWqq) / (probTopbWcs + probTopbWqq)', # massCorrTop (bWqq)
                '(massCorrQCDbb*probQCDbb + massCorrQCDcc*probQCDcc + massCorrQCDb*probQCDb + massCorrQCDc*probQCDc + massCorrQCDothers*probQCDothers) / (probQCDbb + probQCDcc + probQCDb + probQCDc + probQCDothers)', # massCorrQCD
                ## ===== regression nodes for non-MD tagger ===== ##
                '(probWithMassTopbWcs + probWithMassTopbWqq + probWithMassTopbWc + probWithMassTopbWs + probWithMassTopbWq) / (probWithMassTopbWcs + probWithMassTopbWqq + probWithMassTopbWc + probWithMassTopbWs + probWithMassTopbWq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers)', # probWithMassTopvsQCD
                '(probWithMassWcs + probWithMassWqq) / (probWithMassWcs + probWithMassWqq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers)', # probWithMassWvsQCD
                '(probWithMassZbb + probWithMassZcc + probWithMassZss + probWithMassZqq) / (probWithMassZbb + probWithMassZcc + probWithMassZss + probWithMassZqq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers)', # probWithMassZvsQCD
            ]
            assert len(output_nodes_expr) == self.num_output_nodes, f"Output nodes mismatch: {len(output_nodes_expr)} != {self.num_output_nodes}"
            
            # calculate output nodes
            output_nodes = []
            for i, expr in enumerate(output_nodes_expr):
                output_nodes.append(eval(get_expr(expr)))
            output_nodes = torch.stack(output_nodes, dim=1)
            return torch.concat([output_nodes, x], dim=1) # append hidden layer x after output nodes

        else:
            raise NotImplementedError(f"Unsupported version: {self.glopart_version}")


    def forward(self, *args):
        x = super().forward(*args[:-1])
        output = self.fc(x)
        aux_output = self.aux_fc(torch.cat([x, args[-1].squeeze(2)], dim=1))

        # do softmax
        output_cls, output_rest = output.split([self.num_nodes_cls, output.size(1) - self.num_nodes_cls], dim=1)
        output_cls = torch.softmax(output_cls, dim=1)
        output = torch.cat([output_cls, output_rest], dim=1)
        aux_output = torch.softmax(aux_output, dim=1)

        output = torch.cat([output, aux_output], dim=1)

        return self.postprocess(x, output)


def get_model(data_config, **kwargs):
    assert 'fc_params' in kwargs and 'aux_fc_params' in kwargs, "fc_params and aux_fc_params must be provided in exporter"
    assert 'num_nodes' in kwargs and 'num_nodes_cls' in kwargs and 'aux_num_nodes' in kwargs, "num_nodes, num_nodes_cls and aux_num_nodes must be provided in exporter"
    assert 'num_output_nodes' in kwargs, "num_output_nodes must be provided in exporter"
    assert 'version' in kwargs, "version must be provided in exporter"

    # use SwiGLU-default setup
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=None,
        # network configurations
        pair_input_type='pp',
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pair_norm=False,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={},
        # fc_params=(),
        activation='gelu',
        # GloParT exporter configurations
        num_nodes=None,
        num_nodes_cls=None,
        aux_num_nodes=None,
        fc_params=None,
        aux_fc_params=None,
        num_output_nodes=None,
        version=None,
        input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
        # misc
        trim=True,
        for_inference=False,
    )

    kwargs.pop('loss_gamma', None)
    kwargs.pop('loss_split_reg', False)
    kwargs.pop('loss_composed_split_reg', None)
    kwargs.pop('three_coll', False) # v2 setup, not used for v3
    if kwargs.pop('use_swiglu_config', False):
        cfg.update(
            block_params={"scale_attn_mask": True, "scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
            cls_block_params={"scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        )
    if kwargs.pop('use_pair_norm_config', False):
        cfg.update(
            use_pair_norm=True,
            pair_input_dim=6,
        )

    cfg.update(**kwargs)
    model = GlobalParticleTransformerExporter(**cfg)

    _logger.info('Model config: %s' % str(cfg))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }

    return model, model_info
