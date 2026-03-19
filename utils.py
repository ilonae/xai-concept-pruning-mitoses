import torch
import collections
import collections.abc
import torchvision.transforms as T
from itertools import islice
from crp.concepts import ChannelConcept
from crp.attribution import CondAttribution,AttributionGraph
from crp.helper import get_layer_names,abs_norm
from crp.graph import trace_model_graph
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
import sklearn.metrics as skmetrics


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            #print(param_name, param)
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    name = "model_ft"
    for module_name, module in model._modules[name].named_modules():

        if isinstance(module, torch.nn.Conv2d):
            

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements


        elif isinstance(module, torch.nn.Linear):


            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
    print(num_zeros)
    print(num_elements)

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def create_classification_report(model, device, test_loader):

    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = skmetrics.classification_report(
        y_true=y_true, y_pred=y_pred,  output_dict=True)

    return classification_report

def init_attribution_vars(model):
    transform = T.Compose([T.ToTensor()])
    cc = ChannelConcept()
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    attribution = CondAttribution(model) 
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    return transform, cc, composite, attribution,layer_names

""" def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))

 """
""" def return_non_empty(my_dict):
    temp_dict = {}
    for k, v in my_dict.items():
        if v:
            if isinstance(v, dict):
                return_dict = return_non_empty(v)
                if return_dict:
                    temp_dict[k] = return_dict
                else:
                    temp_dict[k] = v
    return temp_dict
 """

def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, collections.abc.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield key, inner_key, inner_value
        else:
            yield key, value


def get_max_ids(test_dict: dict):
    max_vote_ids={}
    while sum(len(lst) for lst in max_vote_ids.values())<10:
        
        for layer in test_dict:
            max_votes =  take(2, test_dict[layer].items())
            max_vote_ids[layer].append(k_inner)
            print(max_votes, layer)
                
    return max_vote_ids

