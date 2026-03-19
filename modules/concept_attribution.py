
import os
from modules.concept_visualization import plot_imgs
import torch

import zennit.image as zimage
import numpy as np
from PIL import Image
from operator import itemgetter
from modules.utils import init_attribution_vars, nested_dict_iter
from crp.helper import abs_norm



def latent_attribute_imgs(path, iteration,is_pruned,folder, model,device, descending=True):
    condition = [{'y': [1]}]
    concept_atlas={}
    transform, cc, composite, attribution,layer_names=init_attribution_vars(model)
 
    for file in os.listdir(path):
        heatmap_dict={}
        path_to_file = os.path.join(path,file)
        splitpath = path_to_file.rsplit('/', 1)[-1]
        splitname = splitpath.rsplit(".",1)[0]
        image = Image.open(path_to_file).convert('RGB')
        sample = transform(image).unsqueeze(0)
        sample.requires_grad = True
        sample = sample.to(device)
        attr = attribution(sample, condition, composite, record_layer=layer_names)
        torch.cuda.empty_cache()
        
        for model_layer in layer_names:
            name = "model_ft"
            single_layer = model_layer[model_layer.index('.') + 1 : ]
            for module_name, _ in model._modules[name].named_modules():
                if single_layer == module_name:
                    rel_c = cc.attribute(attr.relevances[model_layer], abs_norm=True)
                    concept_ids = torch.argsort(rel_c[0], descending=descending)[:10]
                    conditions = [{model_layer: [id], 'y': [1]} for id in concept_ids]
                    #if iteration >8:
                    #    heatmap, _, _, _ = attribution(sample, conditions, composite)
                    #    heatmaps=(zimage.imgify(heatmap, symmetric=True, grid=(1, len(concept_ids)))) 
                    #    heatmap_dict[module_name]=heatmaps
                    
                    del conditions
                    
                    for concept in concept_ids:
                        concept = concept.cpu().item()
                        if ( (single_layer not in concept_atlas)
                           or
                           (concept not in concept_atlas[single_layer])) :
                            concept_atlas.setdefault(single_layer, dict())[concept] =1
                        else:
                            concept_atlas[single_layer][concept]+=1 
                            
                    del concept_ids
                    torch.cuda.empty_cache() 
                    
        del attr
        del sample
        torch.cuda.empty_cache() 
        #if iteration >8:
        #    plot_imgs(iteration,is_pruned,heatmap_dict,image,splitname,folder)
    return concept_atlas

def local_attribute_imgs(path, iteration,is_pruned,folder, model,device, descending=True):
    concept_atlas={}
    features = [64, 64,128, 256, 256,512,512, 512, 4096,4096, 4096]
    transform, cc, composite, attribution,layer_names=init_attribution_vars(model)
    for ind,single_layer in enumerate(layer_names):
        layer_concept_dict={}
        concept_attrs={}
        conditions = [{single_layer: [id], 'y': [1]} for id in np.arange(0, features[ind])]
        for file in os.listdir(path):
            imgs={}
            mask = torch.zeros(512, 512)
            mask[200:300, 200:300] = 1
            path_to_file = os.path.join(path,file)
            image = Image.open(path_to_file).convert('RGB')
            sample = transform(image).unsqueeze(0)
            sample.requires_grad = True
            sample = sample.to(device)
            rel_c = []
            for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=2):
                masked = attr.heatmap * mask[None, :, :]
                rel_c.append(torch.sum(masked, dim=(1, 2)))
            rel_c = torch.cat(rel_c)

            indices = torch.argsort(rel_c, descending=True)[:5]
            percentages = indices, abs_norm(rel_c)[indices]*100
            for idx, concept in enumerate(indices):
                concept = concept.cpu().item()
                if ( (concept not in layer_concept_dict)) :
                    layer_concept_dict[concept] =1
                else:
                    layer_concept_dict[concept]+=1 
                    
        layer_concept_dict= dict( sorted(layer_concept_dict.items(), key=itemgetter(1),reverse=True))
        layer_concept_dict  = {k: layer_concept_dict[k] for k in list(layer_concept_dict)[:5]}
        concept_atlas[single_layer] = layer_concept_dict
        print(single_layer,layer_concept_dict)
        torch.cuda.empty_cache()
        
    return concept_atlas

def retrieve_prunable_true_concepts(iteration,is_pruned, model, device):
    prunable_channels_dict={}
    true_images_path='examples_ds_from_MP.train.HTW.train/True'
    true_concept_atlas = latent_attribute_imgs(true_images_path, iteration,is_pruned,
     'outputs/true_attributions', model, device, False)

    torch.cuda.empty_cache()
    irrelevant_channels_dict={}
    most_irrelevant_channels_dict={}
    prunable_channels_dict ={}
    
    for layer in true_concept_atlas:
        #print(false_concept_atlas[layer])
        irrelevant_channels_dict[layer] = {k: true_concept_atlas[layer][k] for k in true_concept_atlas[layer]}   
        #print(irrelevant_channels_dict[layer])
        irrelevant_channels_dict[layer] = dict( sorted(irrelevant_channels_dict[layer].items(),
                                                       key=itemgetter(1),reverse=True))
        #print(irrelevant_channels_dict)
      
    del true_concept_atlas
    torch.cuda.empty_cache()
    irrelevant_channels_dict_lst = list(nested_dict_iter((irrelevant_channels_dict)))
    #print(irrelevant_channels_dict_lst)
    sorting_record = sorted(irrelevant_channels_dict_lst, key = lambda i: i[2], reverse = True)[0:5]
    
    for item in sorting_record:
        if (item[0] not in prunable_channels_dict):
            prunable_channels_dict[item[0]] =[item[1]]
        else:
            prunable_channels_dict[item[0]].append(item[1])
         
    #for layer in irrelevant_channels_dict:
        
        #most_irrelevant_channels_dict[layer] = take(2, irrelevant_channels_dict[layer].items())
        #print(most_irrelevant_channels_dict[layer])
        #prunable_channels[layer] = list(most_irrelevant_channels_dict[layer].keys())

    return prunable_channels_dict

def retrieve_prunable_concept_diff(iteration,is_pruned, model, device):
    prunable_channels_dict={}
    true_images_path='examples_ds_from_MP.train.HTW.train/True'
    false_images_path='examples_ds_from_MP.train.HTW.train/False'

    true_concept_atlas = latent_attribute_imgs(true_images_path, iteration,is_pruned,
     'outputs/true_attributions', model, device, False)
    false_concept_atlas=latent_attribute_imgs(false_images_path,iteration,is_pruned,
    'outputs/false_attributions', model, device, True)
    #print(false_concept_atlas)
    torch.cuda.empty_cache()
    irrelevant_channels_dict={}
    most_irrelevant_channels_dict={}
    prunable_channels_dict ={}
    
    for layer in false_concept_atlas:
        #print(false_concept_atlas[layer])
        final_dict = dict(true_concept_atlas[layer].items() & false_concept_atlas[layer].items())
        #print ("final dictionary", str(final_dict))
        irrelevant_channels_dict[layer] = {k: false_concept_atlas[layer][k] for k in false_concept_atlas[layer]
                                      if k not in final_dict}   
        #print(irrelevant_channels_dict[layer])
        irrelevant_channels_dict[layer] = dict( sorted(irrelevant_channels_dict[layer].items(),
                                                       key=itemgetter(1),reverse=True))
        print(irrelevant_channels_dict)
      
    del true_concept_atlas
    del false_concept_atlas
    torch.cuda.empty_cache()
    irrelevant_channels_dict_lst = list(nested_dict_iter((irrelevant_channels_dict)))
    print(irrelevant_channels_dict_lst)
    sorting_record = sorted(irrelevant_channels_dict_lst, key = lambda i: i[2], reverse = True)[0:10]
    
    for item in sorting_record:
        if (item[0] not in prunable_channels_dict):
            prunable_channels_dict[item[0]] =[item[1]]
        else:
            prunable_channels_dict[item[0]].append(item[1])
         
    #for layer in irrelevant_channels_dict:
        
        #most_irrelevant_channels_dict[layer] = take(2, irrelevant_channels_dict[layer].items())
        #print(most_irrelevant_channels_dict[layer])
        #prunable_channels[layer] = list(most_irrelevant_channels_dict[layer].keys())

    return prunable_channels_dict
