
import os
import torch
import seaborn as sns
import pandas as pd
from modules.utils import init_attribution_vars
import zennit.image as zimage
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import font_manager
import matplotlib.pyplot as plt



font = font_manager.FontProperties(family='sans-serif', weight='bold')
fontfile = font_manager.findfont(font)


def print_classification(report):
    rep = pd.DataFrame(report)
    classres = sns.heatmap(rep.iloc[:-1, :].T, annot=True)
        
    figure = classres.get_figure()    
    figure.savefig('class_report.png', dpi=400)

def gen_heatmaps_concepts(prunable_channels,iteration,device,model):
    transform, _, composite, attribution,_= init_attribution_vars(model)
    path = 'examples_ds_from_MP.train.HTW.train/True'
    for file in os.listdir(path):
        heatmaps={}
        path_to_file = os.path.join(path,file)
        splitpath = path_to_file.rsplit('/', 1)[-1]
        splitname = splitpath.rsplit(".",1)[0]
        image = Image.open(path_to_file).convert('RGB')
        sample = transform(image).unsqueeze(0)
        sample.requires_grad = True
        sample = sample.to(device)
        heatmaps = {}
        for layer in prunable_channels:
            conditions = [{'model_ft.'+layer: [torch.tensor(id).to(device)], 'y': [1]} for id in prunable_channels[layer]]
            if len(conditions):
                heatmap, _, _, _ = attribution(sample, conditions, composite)
                img=(zimage.imgify(heatmap, symmetric=True, grid=(1, len(prunable_channels[layer])))) 
                heatmaps[layer]=img
            del conditions
            
        del sample
        torch.cuda.empty_cache()
        plot_imgs(iteration,False,heatmaps,image,splitname,'irrelevant_concepts')
    del attribution

def gen_heatmaps_starting_layer(concept_atlas,path,model,device,iteration):
    transform, _, composite, attribution,_= init_attribution_vars(model)
    for file in os.listdir(path):
        heatmaps={}
        path_to_file = os.path.join(path,file)
        image = Image.open(path_to_file).convert('RGB')
        sample = transform(image).unsqueeze(0)
        sample.requires_grad = True
        sample = sample.to(device)
        splitpath = path_to_file.rsplit('/', 1)[-1]
        splitname = splitpath.rsplit(".",1)[0]
        
        for layer in concept_atlas:
            conditions = [{layer: [id]} for id in concept_atlas[layer].keys()]
            heatmap, _, _, _ = attribution(sample, conditions, composite, start_layer=layer)
            img = zimage.imgify(heatmap, symmetric=True, grid=(1, len(concept_atlas[layer].keys())))
            heatmaps[layer]=img
            del conditions
            
        del sample
        torch.cuda.empty_cache()
        plot_imgs(iteration,False,heatmaps,image,splitname,'backward_concepts')

def vis_broadcast_heatmaps(concept_atlas,path,model,device,iteration):
    transform, _, composite, attribution,_= init_attribution_vars(model)
    for file in os.listdir(path):
        imgs={}
        
        path_to_file = os.path.join(path,file)
        splitpath = path_to_file.rsplit('/', 1)[-1]
        splitname = splitpath.rsplit(".",1)[0]
        image = Image.open(path_to_file).convert('RGB')
        sample = transform(image).unsqueeze(0)
        sample.requires_grad = True
        sample = sample.to(device)
        
        for layer in concept_atlas:
            conditions =  [{layer: [35], "y": [next(iter(concept_atlas[layer].keys()))]}, {layer: [next(iter(concept_atlas[layer].keys()))], "y": [0]}]
            heatmaps, _, _, _ = attribution(sample, conditions, composite)
            img = zimage.imgify(heatmaps, symmetric=True, grid=(1, len(heatmaps)))
            imgs[layer]=img
            del conditions
            
        del sample
        torch.cuda.empty_cache()
        plot_imgs(iteration,False,heatmaps,image,splitname,'broadcast_concepts')


def plot_imgs(iteration,pruned_bool,heatmaps,sample,samplename,path):
    fontsize=1
    img_fraction=0.2
    font = ImageFont.truetype(fontfile, fontsize)
    
    if pruned_bool:
        flag = '_after_pruning_'
    else:
        flag = '_before_pruning_'
        
    min_img_width = min(i.width for i in heatmaps.values())
    total_height = 0
    heatmap=0
    for i, key in enumerate(heatmaps.copy()):
        heatmap = heatmaps[key]
        heatmap.convert('RGB')
        heatmap = ImageOps.expand(heatmap, border=int(0.2*heatmap.size[1]), fill=(255,255,255))
        while font.getbbox(key)[2] - font.getbbox(key)[0] < img_fraction*heatmap.size[0]:
            # iterate until the text size is just larger than the criteria
            fontsize += 1
            font = ImageFont.truetype(fontfile, fontsize)
        fontsize -= 1
        font = ImageFont.truetype(fontfile, fontsize)
        d = ImageDraw.Draw(heatmap)
        bbox = font.getbbox(key)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((int((heatmap.width-w)/2),0), key, fill=(0,0,0),font=font)
        if heatmap.width > min_img_width:
            heatmaps[key] = heatmap.resize((min_img_width, int(heatmap.height / heatmap.width * min_img_width)), Image.LANCZOS)
        total_height += heatmaps[key].height
        

    wpercent = (min_img_width/float(sample.size[0]))
    hsize = int((float(sample.size[1])*float(wpercent)))
    sample = sample.resize((min_img_width,hsize), Image.LANCZOS)
    
    sample = ImageOps.expand(sample,  border=int(0.1*sample.size[1]), fill=(255,255,255))
    d = ImageDraw.Draw(sample)
    bbox = font.getbbox("Original")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((int((sample.width-w)/2),0), "Original", fill=(0,0,0),font=font)
    
    img_merge = Image.new(heatmap.mode, (min_img_width+sample.width+(2*int(0.1*sample.size[1])), total_height)).convert('RGB')
    img_merge.paste((255,255,255), [0,0,min_img_width+sample.width+2*int(0.1*sample.size[1]),total_height])
    y = 0
    for image in heatmaps.values():
        img_merge.paste(image, (0, y))

        y += image.height
    img_merge.paste(sample, (sample.width, int(sample.height/2)-2*int(0.1*sample.size[1])))
    img_merge.save(path+'/iteration_'+str(iteration)+str(flag)+samplename+'.jpg')
