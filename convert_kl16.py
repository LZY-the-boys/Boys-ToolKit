import argparse
import io

import requests
import torch
import yaml

from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)


def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    old_keys = ["encoder.down.4.attn.0.norm.weight","encoder.down.4.attn.0.norm.bias","encoder.down.4.attn.0.q.weight","encoder.down.4.attn.0.q.bias","encoder.down.4.attn.0.k.weight","encoder.down.4.attn.0.k.bias","encoder.down.4.attn.0.v.weight","encoder.down.4.attn.0.v.bias","encoder.down.4.attn.0.proj_out.weight","encoder.down.4.attn.0.proj_out.bias","encoder.down.4.attn.1.norm.weight","encoder.down.4.attn.1.norm.bias","encoder.down.4.attn.1.q.weight","encoder.down.4.attn.1.q.bias","encoder.down.4.attn.1.k.weight","encoder.down.4.attn.1.k.bias","encoder.down.4.attn.1.v.weight","encoder.down.4.attn.1.v.bias","encoder.down.4.attn.1.proj_out.weight","encoder.down.4.attn.1.proj_out.bias","decoder.up.4.attn.0.norm.weight","decoder.up.4.attn.0.norm.bias","decoder.up.4.attn.0.q.weight","decoder.up.4.attn.0.q.bias","decoder.up.4.attn.0.k.weight","decoder.up.4.attn.0.k.bias","decoder.up.4.attn.0.v.weight","decoder.up.4.attn.0.v.bias","decoder.up.4.attn.0.proj_out.weight","decoder.up.4.attn.0.proj_out.bias","decoder.up.4.attn.1.norm.weight","decoder.up.4.attn.1.norm.bias","decoder.up.4.attn.1.q.weight","decoder.up.4.attn.1.q.bias","decoder.up.4.attn.1.k.weight","decoder.up.4.attn.1.k.bias","decoder.up.4.attn.1.v.weight","decoder.up.4.attn.1.v.bias","decoder.up.4.attn.1.proj_out.weight","decoder.up.4.attn.1.proj_out.bias","decoder.up.4.attn.2.norm.weight","decoder.up.4.attn.2.norm.bias","decoder.up.4.attn.2.q.weight","decoder.up.4.attn.2.q.bias","decoder.up.4.attn.2.k.weight","decoder.up.4.attn.2.k.bias","decoder.up.4.attn.2.v.weight","decoder.up.4.attn.2.v.bias","decoder.up.4.attn.2.proj_out.weight","decoder.up.4.attn.2.proj_out.bias"]
    for k in old_keys:
        if vae_state_dict[k].shape == torch.Size([512, 512, 1,1 ]):
            # [512, 512, 1, 1] -> [512, 512]
            vae_state_dict[k] = vae_state_dict[k].squeeze(-1).squeeze(-1)
        if 'up' not in k:
            new_checkpoint[
                k.replace('attn', 'attentions').replace('norm', 'group_norm').replace('.q.', '.to_q.').replace('.k.', '.to_k.').replace('.v.', '.to_v.').replace('proj_out', 'to_out').replace('.down.', '.down_blocks.')
            ] = vae_state_dict[k]
        else:
            _k = k.replace('attn', 'attentions').replace('norm', 'group_norm').replace('.q.', '.to_q.').replace('.k.', '.to_k.').replace('.v.', '.to_v.').replace('proj_out', 'to_out').replace('.up.', '.up_blocks.')
            _ks = _k.split('.')
            _k = '.'.join(_ks[0:2] + [str(4-int(_ks[2]))] + _ks[3:])
            new_checkpoint[
                _k
            ] = vae_state_dict[k]
    
    for k in ["encoder.down_blocks.4.attentions.0.to_out.weight", "encoder.down_blocks.4.attentions.0.to_out.bias", "encoder.down_blocks.4.attentions.1.to_out.weight", "encoder.down_blocks.4.attentions.1.to_out.bias", "decoder.up_blocks.0.attentions.0.to_out.weight", "decoder.up_blocks.0.attentions.0.to_out.bias", "decoder.up_blocks.0.attentions.1.to_out.weight", "decoder.up_blocks.0.attentions.1.to_out.bias", "decoder.up_blocks.0.attentions.2.to_out.weight", "decoder.up_blocks.0.attentions.2.to_out.bias"]:
        ks = k.split('.')
        new_checkpoint['.'.join(ks[:-1] +['0'] + [ks[-1]])] = new_checkpoint[k]
        del new_checkpoint[k]
    # for i in range(num_down_blocks):
    #     for c in ['qkv']:
    #         new_checkpoint[f"encoder.down.{i}.attn.{}"] = vae_state_dict.pop(
    #             f"encoder.down.{i}.downsample.conv.bias"
    #         )
    #         meta_path = {"old": f"attn.{i}.{c}", "new": f"attentions.{i}.to_{c}"}
    #         assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    #     meta_path = {"old": f"attn.{i}.proj_out", "new": f"attentions.{i}.to_out"}
    #     assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for k in old_keys:
        if k in new_checkpoint:
            del new_checkpoint[k]
    return new_checkpoint


def vae_pt_to_vae_diffuser(
    checkpoint_path: str,
    output_path: str,
):
    # Only support V1
    # r = requests.get(
    #     " https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    # )
    # io_obj = io.BytesIO(r.content)

    # original_config = yaml.safe_load(io_obj)
    original_config = yaml.safe_load(open('kl-16.yaml','r'))
    image_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]
    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    vae_config = {
        'sample_size': 256,
        'in_channels': 3,
        'out_channels': 3,
        'down_block_types': 
        ('DownEncoderBlock2D',
        'DownEncoderBlock2D',
        'DownEncoderBlock2D',
        'DownEncoderBlock2D',
        'AttnDownEncoderBlock2D'),
        'up_block_types': 
        ('AttnUpDecoderBlock2D',
        'UpDecoderBlock2D',
        'UpDecoderBlock2D',
        'UpDecoderBlock2D',
        'UpDecoderBlock2D',),
        'block_out_channels': (128, 128, 256, 256, 512),
        'latent_channels': 16,
        'layers_per_block': 2
    }
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--vae_pt_path", default=None, type=str, required=True, help="Path to the VAE.pt to convert.")
        parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the VAE.pt to convert.")

        args = parser.parse_args()

        vae_pt_to_vae_diffuser(args.vae_pt_path, args.dump_path)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
