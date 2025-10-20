from detectron2.utils.registry import Registry

AGGREGATION_REGISTRY = Registry("AGGREGATION")  #在初始化的时候就创建了一个注册器，然后用装饰器的方式把几个类存到这个注册器中，然后用字符串就能返回对应的类
AGGREGATION_REGISTRY.__doc__ = """
Registry for cost aggregation, which estimate aggregated cost volume from images
The registered object must be a callable that accepts one arguments:
1. A :class:`detectron2.config.CfgNode`
Registered object must return instance of :class:`nn.Module`.
"""


def build_aggregation(cfg):
    """
    Build a cost aggregation predictor from `cfg.MODEL.AGGREGATION.NAME`.
    Returns:
        an instance of :class:`nn.Module`
    """

    aggregation_name = cfg.MODEL.AGGREGATION.NAME
    aggregation_predictor = AGGREGATION_REGISTRY.get(aggregation_name)(cfg)
    return aggregation_predictor