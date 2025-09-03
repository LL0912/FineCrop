import inspect
from SITS.apis.base_trainer import BaseTrainer


def build_from_cfg(cfg, registry):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` or must contain the key "type", but got {type(registry)}')

    args = cfg.copy()

    obj_type = args.pop('type')


    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    if issubclass(obj_cls, BaseTrainer):
        return obj_cls
        
    if hasattr(obj_cls, 'from_config') and inspect.ismethod(getattr(obj_cls, 'from_config')):
        # 调用 from_config 生成构造参数
        constructor_args = obj_cls.from_config(cfg)
    else:
        # 直接使用 params 参数
        constructor_args = args.get('params', {})
    return obj_cls(**constructor_args)
    # else:
    #     print(args['params'])
    #     return obj_cls(**args['params'])
