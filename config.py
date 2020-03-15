# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2018-12-01
# --------------------------------------------------------
class Config(object):
    
    NAME= "ilsvrc_xeception"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    CH_CFG=[[8,48,96],
            [240,144,288],
            [240,144,288]]
