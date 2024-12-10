import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd
from model import myUNet2DConditionModel, my_vae_decode, _my_decode, my_vae_encode
from my_utils.training_utils import SD_TURBO_PATH

import logging
logging.basicConfig(filename='report.log', level=logging.DEBUG)
print = logging.debug


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(SD_TURBO_PATH, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(SD_TURBO_PATH, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained(SD_TURBO_PATH, subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decode = my_vae_decode.__get__(vae, vae.__class__)
        vae.encode = my_vae_encode.__get__(vae, vae.__class__)
        vae._decode = _my_decode.__get__(vae, vae.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
        vae.decoder.conv128 = torch.nn.Conv2d(512, 1, kernel_size=3, padding=1).cuda()
        vae.decoder.conv256 = torch.nn.Conv2d(512, 1, kernel_size=3, padding=1).cuda()
        vae.decoder.conv512 = torch.nn.Conv2d(128, 1, kernel_size=3, padding=1).cuda()

        vae.decoder.conv_0 = torch.nn.Conv2d(320, 512, kernel_size=3, padding=1).cuda()
        vae.decoder.conv_1 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda()
        vae.decoder.conv_2 = torch.nn.Conv2d(512, 128, kernel_size=3, padding=1).cuda()
        vae.decoder.conv_3 = torch.nn.Conv2d(512, 128, kernel_size=3, padding=1).cuda()

        vae.encoder.conv256 = torch.nn.Conv2d(32, 128, kernel_size=3, padding=1).cuda()
        vae.encoder.conv128 = torch.nn.Conv2d(64, 256, kernel_size=3, padding=1).cuda()
        vae.encoder.conv64 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1).cuda()

        unet = myUNet2DConditionModel.from_pretrained(SD_TURBO_PATH, subfolder="unet")
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.unet_layer_0_1 = torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1).cuda()
        self.unet_layer_1_2 = torch.nn.Conv2d(1280, 640, kernel_size=3, padding=1).cuda()
        self.unet_layer_2_3 = torch.nn.Conv2d(640, 320, kernel_size=3, padding=1).cuda()
        self.conv16 = torch.nn.Conv2d(1280, 1, kernel_size=3, padding=1).cuda()
        self.conv32 = torch.nn.Conv2d(1280, 1, kernel_size=3, padding=1).cuda()
        self.conv64 = torch.nn.Conv2d(320, 1, kernel_size=3, padding=1).cuda()

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            self.conv16.load_state_dict(sd['unet_conv_state_dict']['conv16'])
            self.conv32.load_state_dict(sd['unet_conv_state_dict']['conv32'])
            self.conv64.load_state_dict(sd['unet_conv_state_dict']['conv64'])
            self.unet_layer_0_1.load_state_dict(sd['unet_conv_state_dict']['unet_layer_0_1'])
            self.unet_layer_1_2.load_state_dict(sd['unet_conv_state_dict']['unet_layer_1_2'])
            self.unet_layer_2_3.load_state_dict(sd['unet_conv_state_dict']['unet_layer_2_3'])
            vae.decoder.conv128.load_state_dict(sd['vae_conv_state_dict']['conv128'])
            vae.decoder.conv256.load_state_dict(sd['vae_conv_state_dict']['conv256'])
            vae.decoder.conv512.load_state_dict(sd['vae_conv_state_dict']['conv512'])
            vae.decoder.conv_0.load_state_dict(sd['vae_conv_state_dict']['conv_0'])
            vae.decoder.conv_1.load_state_dict(sd['vae_conv_state_dict']['conv_1'])
            vae.decoder.conv_2.load_state_dict(sd['vae_conv_state_dict']['conv_2'])
            vae.decoder.conv_3.load_state_dict(sd['vae_conv_state_dict']['conv_3'])
            vae.encoder.conv256.load_state_dict(sd['vae_conv_state_dict']['conv256_e'])
            vae.encoder.conv128.load_state_dict(sd['vae_conv_state_dict']['conv128_e'])
            vae.encoder.conv64.load_state_dict(sd['vae_conv_state_dict']['conv64_e'])

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            torch.nn.init.constant_(vae.encoder.conv256.weight, 1e-5)
            torch.nn.init.constant_(vae.encoder.conv128.weight, 1e-5)
            torch.nn.init.constant_(vae.encoder.conv64.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        self.unet_layer_0_1.requires_grad_(True)
        self.unet_layer_1_2.requires_grad_(True)
        self.unet_layer_2_3.requires_grad_(True)
        self.conv16.requires_grad_(True)
        self.conv32.requires_grad_(True)
        self.conv64.requires_grad_(True)
        self.vae.decoder.conv128.requires_grad_(True)
        self.vae.decoder.conv256.requires_grad_(True)
        self.vae.decoder.conv512.requires_grad_(True)
        self.vae.decoder.conv_0.requires_grad_(True)
        self.vae.decoder.conv_1.requires_grad_(True)
        self.vae.decoder.conv_2.requires_grad_(True)
        self.vae.decoder.conv_3.requires_grad_(True)
        self.vae.encoder.conv256.requires_grad_(True)
        self.vae.encoder.conv128.requires_grad_(True)
        self.vae.encoder.conv64.requires_grad_(True)

    def forward(self, c_t, c_t_embed, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        if deterministic:
            encoded_control = self.vae.encode(c_t, c_t_embed).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred, up_ft = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,)
            model_pred = model_pred.sample

            temp1 = self.tanh(self.unet_layer_0_1(F.interpolate(up_ft[0], scale_factor=2, mode='bilinear'))) + up_ft[1]
            temp2 = self.tanh(self.unet_layer_1_2(F.interpolate(temp1, scale_factor=2, mode='bilinear'))) + up_ft[2]
            temp3 = self.tanh(self.unet_layer_2_3(temp2)) + up_ft[3]
            mask_16 = self.sigmoid(self.conv16(up_ft[0]))
            mask_32 = self.sigmoid(self.conv32(temp1))
            mask_64 = self.sigmoid(self.conv64(temp3))
            mask_list = [mask_16, mask_32, mask_64]

            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            decode_result = self.vae.decode(x_denoised / self.vae.config.scaling_factor, temp3)
            output_image = decode_result[0].clamp(-1, 1)
            mask_list = mask_list + decode_result[1]

        else:
            print("Not used...")

        return output_image, mask_list

    def save_model(self, outf):
        sd = {}
        unet_conv_state_dict = {
            'conv16': self.conv16.state_dict(), 
            'conv32': self.conv32.state_dict(), 
            'conv64': self.conv64.state_dict(), 
            'unet_layer_0_1': self.unet_layer_0_1.state_dict(), 
            'unet_layer_1_2': self.unet_layer_1_2.state_dict(), 
            'unet_layer_2_3': self.unet_layer_2_3.state_dict(), 
        }
        vae_conv_state_dict = {
            'conv128': self.vae.decoder.conv128.state_dict(), 
            'conv256': self.vae.decoder.conv256.state_dict(), 
            'conv512': self.vae.decoder.conv512.state_dict(), 
            'conv_0': self.vae.decoder.conv_0.state_dict(), 
            'conv_1': self.vae.decoder.conv_1.state_dict(), 
            'conv_2': self.vae.decoder.conv_2.state_dict(), 
            'conv_3': self.vae.decoder.conv_3.state_dict(), 
            'conv256_e': self.vae.encoder.conv256.state_dict(), 
            'conv128_e': self.vae.encoder.conv128.state_dict(), 
            'conv64_e': self.vae.encoder.conv64.state_dict(), 
        }
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["unet_conv_state_dict"] = unet_conv_state_dict
        sd["vae_conv_state_dict"] = vae_conv_state_dict
        torch.save(sd, outf)
