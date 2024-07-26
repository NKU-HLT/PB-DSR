#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
#-----wsy add-------------------------------------------------
import sys
sys.path.append("[PB-DSR DIR]/")
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 注意 
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")

@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)

def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    #------wsy add-------------------------------------------------------------------------------------------------
    # # 注意： 这里是debug需要的：（实际使用指令训练时需要注释掉
    # cfg1 = OmegaConf.load("[PB-DSR DIR]/examples/hubert/config/finetune/base_10h.yaml")
    # # cfg2 = OmegaConf.load("[PB-DSR DIR]/.hydra/config.yaml")

    # cfg2=OmegaConf.load("[PB-DSR DIR]/wsy/private/training_log/7.23_cdsd_baseline/.hydra/config.yaml") # 使用这个可以解决model的问题
    # cfg3=OmegaConf.load("[PB-DSR DIR]/wsy/private/training_log/7.23_cdsd_baseline/.hydra/hydra.yaml")

    # cfg = OmegaConf.merge(cfg2,cfg1,cfg3)
    # cfg.optimization.stop_min_lr=-1.0
    # # """
    # # 报错：omegaconf.errors.MissingMandatoryValue: Missing mandatory value: model
    # #     但是cfg1中有model值，cfg2中没有，最后merge得到的cfg也没有model
    # #     查到的解释：OmegaConf.merge()也是以最后输入的config为配置值的最终设置，相当于“后来为先”原则，即时间最顺序最后一个config优先级最高。 
    # #     OmegaConf.merge()的更新还具有顺序保持的特点，即在参数顺序上时依照“先来为序”的原则，会按照config1的顺序对参数进行排列。
    # # """
    #--------------------------------------------------------------------------------------------------------------
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    #---wsy fix---------------------------------------------------------
    # try:
    #     if cfg.common.profile:
    #         with torch.cuda.profiler.profile():
    #             with torch.autograd.profiler.emit_nvtx():
    #                 distributed_utils.call_main(cfg, pre_main, **kwargs)
    #     else:
    #         distributed_utils.call_main(cfg, pre_main, **kwargs)
    # except BaseException as e:
    #     if not cfg.common.suppress_crashes:
    #         raise
    #     else:
    #         logger.error("Crashed! " + str(e))
    distributed_utils.call_main(cfg, pre_main, **kwargs)
    #---------------------------------------------------------------

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val


def cli_main():
    try:
        from hydra._internal.utils import get_args
        #------wsy fix--------------------------------
        # cfg_name = get_args().config_name or "config"
        cfg_name = get_args().config_name or "base_10h"
        #----------------------------------------------
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
