import torch
import torch.nn.utils.prune as prune
from torch import nn


def prune_selected_concepts(max_vote_concept_dict,model):
    for key,value in max_vote_concept_dict.items():
        name = "model_ft"
        for module_name, module in model._modules[name].named_modules():
            layer_nr = int(key.split('.', 1)[1])
            mod = key.split('.', 1)[0]
            if mod == module_name:
                mask_tensor = torch.ones(module[layer_nr].weight.shape, device=module[layer_nr].weight.device)
                for concept in value:
                    #pruning differences between Conv2d and Linear layer
                    if isinstance(module[layer_nr], nn.Conv2d):
                        #masking channel by id so nothing is passed further
                        mask_tensor[concept, :, :, :] = torch.zeros(module[layer_nr].weight[0].shape)
                    elif isinstance(module[layer_nr], nn.Linear):
                        mask_tensor[concept,:] = torch.zeros(module[layer_nr].weight[0].shape)
                m = prune.custom_from_mask(module[layer_nr], name='weight', mask=mask_tensor)
                #print(mask_tensor)
                print(m)
                prune.remove(module[layer_nr],"weight")
            torch.cuda.empty_cache()
    return model
