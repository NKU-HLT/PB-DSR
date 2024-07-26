# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools as it
from typing import Any, Dict, List

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqModel

#--------wsy add-------------------
from wsy import hparam as hp 
import pickle as pkl
import os
#----------------------------------

class BaseDecoder:
    def __init__(self, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()

    def generate(
        self, models: List[FairseqModel], sample: Dict[str, Any], **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        #--------------wsy add------------------------------------------------------------------------
        if hp.prepare_datastore: # 注意
            pkl_path=os.path.join(hp.o_datastore_dir,str(sample["id"].detach().cpu().numpy()[0])+".pkl")
            with open(pkl_path, 'wb') as f:
                # pkl.dump(emissions["encoder_out"].squeeze(), f) # 实际保存的是hubert feature
                pkl.dump(emissions["hubert_feature"].squeeze(), f) # 实际保存的是经过投影头的特征，如果有投影头的话
            return None
        #--------wsy fix------------------------------------------------------------------------------------
        # return self.decode(emissions)
        if hp.knn_aug:
            return self.decode(emissions[0],emissions[1]["hubert_feature"])
        else:
            return self.decode(emissions)
        #---------------------------------------------------------------

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]
        encoder_out = model(**encoder_input)
        #-----wsy add---------------
        if hp.prepare_datastore:
            return encoder_out
        #---------------------------
        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out)
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError
