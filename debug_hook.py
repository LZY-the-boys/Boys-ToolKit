import torch
import logging

logger = logging.getLogger('cruise')

max_outliner = 0
import warnings

def debug_nan(model):

    class nan_hook:

        def __init__(self,name, module):
            # 注入 name 信息
            self.name=name
            module.register_forward_hook(self._hook)

        def _hook(self, module, inp, output):
            # 带lora的时候不准
            # printf(self.name)
            
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):

                if out is None:
                    continue
                if self.name == 'model':
                    # dataclass
                    continue
                if isinstance(out, dict):
                    # for k,v in out.__dict__.items():
                    #     try:
                    #         print(k, v.max())
                    #     except:
                    #         pass
                    return
                # else:
                #     printf(out.max())
                
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    logger.error(f"Found NAN in {self.name} output {i} at indices: ", nan_mask.nonzero())
                    import pdb;pdb.set_trace()
                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    logger.error(f"Found INF in {self.name} output {i} at indices: ", inf_mask.nonzero())
                    import pdb;pdb.set_trace()
                outliner = out.abs().max()
                if outliner > 1000:
                    # raise RuntimeError(f"Found outlier in {self.name} output {out_max}: ", out.argmax())
                    # warnings.warn(f"Found outlier in {self.name} output {out_max}: {out.argmax()}" )
                    global max_outliner
                    max_outliner = max(max_outliner, outliner.item())
                

            # torch.isinf(hidden_states).any()
            # torch.isinf(hidden_states).nonzero()
    
    # for submodule in model.modules():
    for name,submodule in model.named_modules():
        nan_hook(name, submodule)


def get_activation_norm():
    def hook(name, cache, module, input):
        if len(input) == 0:
            return
        act_norm = input[0].detach().double().norm(2, dim=-1).mean().item()
        cache[name] = act_norm
    return hook

from functools import partial

def log_act_hook(model, act_norm_cache):

    class log_hook:

        def __init__(self,name, module, act_norm_cache):
            # 注入 name 信息
            self.name=name
            module.register_forward_pre_hook(
                partial(get_activation_norm(), 'act/'+ name, act_norm_cache)
            )
                
    for name,submodule in model.named_modules():
        log_hook(name, submodule, act_norm_cache)


def check_parameter(model, state_dict=True):
    ans = {}
    if state_dict: 
        p = model.state_dict().items()
    else:
        p = model.named_parameters()
    for name, param in p:
        # ans['mean/' + name] = param.detach().abs().mean().item()
        ans['norm/' + name] = param.detach().norm(2, dim=-1).mean().item()
    return ans