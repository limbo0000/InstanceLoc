# disclaimer: inspired by MoCo official repo and PyContrast repo
import argparse
import pickle as pkl
import sys

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I', help='input model path')
    parser.add_argument('output', metavar='O', help='output path')
    parser.add_argument('--c4', action='store_true', help='using ema model')
    args = parser.parse_args()

    print('=========================')
    print(f'converting {args.input}')
    print('=========================')
    torch_weight = torch.load(args.input)['state_dict']
    new_state_dict = {}

    for k, v in torch_weight.items():
        # C4 backbone treats C5 layers as the shared head
        if 'roi_head.shared_head' in k:
            old_k = k
            k = k.replace('roi_head.shared_head', 'backbone')
            print(old_k, '---->', k)

        if not args.c4 and 'roi_head.bbox_head.shared_fcs' in k:
            old_k = k
            k = k.replace(f'roi_head.bbox_head.shared_fcs.0.weight',
                          f'roi_heads.box_head.fc1.weight')
            k = k.replace(f'roi_head.bbox_head.shared_fcs.0.bias',
                          f'roi_heads.box_head.fc1.bias')
            print(old_k, '---->', k)
            new_state_dict[k] = v.numpy()
            continue

        if 'backbone' in k and 'layer' not in k and 'backbone_k' not in k:
            old_k = k
            if args.c4:
                # C4 stem
                k = k.replace('backbone',
                              'backbone.stem').replace('bn1', 'conv1.norm')
            else:
                # FPN stem
                k = k.replace('backbone', 'backbone.bottom_up.stem').replace(
                    'bn1', 'conv1.norm')
            print(old_k, '---->', k)
        elif 'backbone' in k and 'backbone_k' not in k and 'layer' in k:
            if args.c4:
                # C4 Backbone
                old_k = k
                if 'layer4' not in k:
                    k = k.replace("layer1", "res2")
                    k = k.replace("layer2", "res3")
                    k = k.replace("layer3", "res4")
                else:
                    k = k.replace("backbone", "roi_heads")
                    k = k.replace("layer4", "res5")
                k = k.replace("bn1", "conv1.norm")
                k = k.replace("bn2", "conv2.norm")
                k = k.replace("bn3", "conv3.norm")
                k = k.replace("downsample.0", "shortcut")
                k = k.replace("downsample.1", "shortcut.norm")
            else:
                # FPN backbone
                old_k = k
                k = k.replace('backbone', 'backbone.bottom_up')
                k = k.replace("layer1", "res2")
                k = k.replace("layer2", "res3")
                k = k.replace("layer3", "res4")
                k = k.replace("layer4", "res5")
                k = k.replace("bn1", "conv1.norm")
                k = k.replace("bn2", "conv2.norm")
                k = k.replace("bn3", "conv3.norm")
                k = k.replace("downsample.0", "shortcut")
                k = k.replace("downsample.1", "shortcut.norm")
            print(old_k, '--->', k)
        elif 'neck' in k and 'neck_k' not in k:
            # FPN neck
            old_k = k
            # replace lateral conv
            k = k.replace('neck.lateral_convs.0.bn',
                          'backbone.fpn_lateral2.norm')
            k = k.replace('neck.lateral_convs.1.bn',
                          'backbone.fpn_lateral3.norm')
            k = k.replace('neck.lateral_convs.2.bn',
                          'backbone.fpn_lateral4.norm')
            k = k.replace('neck.lateral_convs.3.bn',
                          'backbone.fpn_lateral5.norm')

            k = k.replace('neck.lateral_convs.0.conv', 'backbone.fpn_lateral2')
            k = k.replace('neck.lateral_convs.1.conv', 'backbone.fpn_lateral3')
            k = k.replace('neck.lateral_convs.2.conv', 'backbone.fpn_lateral4')
            k = k.replace('neck.lateral_convs.3.conv', 'backbone.fpn_lateral5')

            # replace fpn conv
            k = k.replace('neck.fpn_convs.0.bn', 'backbone.fpn_output2.norm')
            k = k.replace('neck.fpn_convs.1.bn', 'backbone.fpn_output3.norm')
            k = k.replace('neck.fpn_convs.2.bn', 'backbone.fpn_output4.norm')
            k = k.replace('neck.fpn_convs.3.bn', 'backbone.fpn_output5.norm')

            k = k.replace('neck.fpn_convs.0.conv', 'backbone.fpn_output2')
            k = k.replace('neck.fpn_convs.1.conv', 'backbone.fpn_output3')
            k = k.replace('neck.fpn_convs.2.conv', 'backbone.fpn_output4')
            k = k.replace('neck.fpn_convs.3.conv', 'backbone.fpn_output5')
            print(old_k, '--->', k)
        elif 'roi_head.bbox_head.shared_convs' in k:
            # 4conv in det head
            old_k = k
            conv_idx = int(k.split('.')[3])
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.conv.weight',
                f'roi_heads.box_head.conv{conv_idx+1}.weight')
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.bn.weight',
                f'roi_heads.box_head.conv{conv_idx+1}.norm.weight')
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.bn.bias',
                f'roi_heads.box_head.conv{conv_idx+1}.norm.bias')
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.bn.running_mean',
                f'roi_heads.box_head.conv{conv_idx+1}.norm.running_mean')
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.bn.running_var',
                f'roi_heads.box_head.conv{conv_idx+1}.norm.running_var')
            k = k.replace(
                f'roi_head.bbox_head.shared_convs.{conv_idx}.bn.num_batches_tracked',
                f'roi_heads.box_head.conv{conv_idx+1}.norm.num_batches_tracked'
            )
            print(old_k, '--->', k)
        else:
            continue
        new_state_dict[k] = v.numpy()

    res = {
        "model": new_state_dict,
        "__author__": "Ceyuan",
        "matching_heuristics": True
    }

    with open(args.output, "wb") as f:
        pkl.dump(res, f)
