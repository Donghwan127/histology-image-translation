import os
import gc
import wandb
import torch
import shutil
import lpips
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import torch.backends.cuda
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import build_feature_extractor

from diffusion_ffpe.my_utils import calculate_fid, evaluate, get_mu_sigma
from diffusion_ffpe.dino_struct import DinoStructureLoss
from diffusion_ffpe.unpaired_dataset import UnpairedDataset, make_dataset
from diffusion_ffpe.model import Diffusion_FFPE, initialize_text_encoder, initialize_discriminator


def parse_args_training():
    parser = argparse.ArgumentParser(description="Script for training Diffusion-FFPE with Diffusion Bridge.")

    # args for the model
    parser.add_argument("--model_path", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for dataset and dataloader options
    parser.add_argument("--train_source_folder", type=str, default="TRAIN_FF_PATH")
    parser.add_argument("--train_target_folder", type=str, default="TRAIN_FFPE_PATH")
    parser.add_argument("--valid_source_folder", type=str, default="VALID_FF_PATH")
    parser.add_argument("--valid_target_folder", type=str, default="VALID_FFPE_PATH")
    parser.add_argument("--fixed_caption_src", type=str, default="frozen section")
    parser.add_argument("--fixed_caption_tgt", type=str, default="paraffin section")
    parser.add_argument("--train_img_prep", type=str, default='no_resize')
    parser.add_argument("--val_img_prep", type=str, default='no_resize')
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device for training.")

    # args for the optimization options
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.000005)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # args for diffusion bridge
    parser.add_argument("--num_bridge_steps", default=4, type=int, help="Number of diffusion steps for bridge")
    parser.add_argument("--bridge_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--lambda_bridge", default=1.0, type=float, help="Weight for bridge matching loss")
    parser.add_argument("--lambda_bridge_lpips", default=5.0, type=float, help="Weight for bridge LPIPS loss")
    
    # args for the loss function
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_ot", default=1.0, type=float, help="Optimal transport loss weight")

    # args for validation and logging
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--tracker_project_name", type=str, default='unpaired_frozen2ffpe_bridge')
    parser.add_argument("--output_dir", type=str, default="./experiments/diffusion-ffpe-bridge")
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--validation_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--validation_num_images", type=int, default=-1)

    # args for other settings
    parser.add_argument("--seed", type=int, default=2024, help="A seed for reproducible training.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="./output/diffusion-ffpe-bridge/checkpoints/")
    parser.add_argument("--max_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    args = parser.parse_args()
    return args


def get_bridge_schedule(num_steps, schedule_type="linear"):
    """Generate bridge schedule alpha_t from 0 to 1"""
    if schedule_type == "linear":
        return torch.linspace(0, 1, num_steps + 1)
    elif schedule_type == "cosine":
        t = torch.linspace(0, 1, num_steps + 1)
        return 1 - torch.cos(t * np.pi / 2)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def bridge_interpolate(x_start, x_end, alpha_t, noise_scale=0.1):
    """
    Interpolate between x_start and x_end with noise injection
    x_t = alpha_t * x_end + (1 - alpha_t) * x_start + noise
    """
    bsz = x_start.shape[0]
    device = x_start.device
    
    # Add small noise for stochastic bridge
    if noise_scale > 0:
        noise = torch.randn_like(x_start) * noise_scale * (alpha_t * (1 - alpha_t)).sqrt()
    else:
        noise = 0
    
    x_t = alpha_t * x_end + (1 - alpha_t) * x_start + noise
    return x_t


def main(args):

    accelerator = Accelerator(
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    )

    weight_dtype = torch.float32
    set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare Dataset
    train_dataset = UnpairedDataset(source_folder=args.train_source_folder, target_folder=args.train_target_folder,
                                    image_prep=args.train_img_prep)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workers)
    l_images_src_test = make_dataset(args.valid_source_folder)
    ref_path = os.path.join(args.output_dir, "valid_statistics.npz")

    # Prepare Text Embeddings
    tokenizer, text_encoder = initialize_text_encoder(args.model_path)
    fixed_a2b_tokens = tokenizer(args.fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length",
                                 truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(args.fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length",
                                 truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del tokenizer, text_encoder

    # Define Model (Bridge networks for both directions)
    model_a2b = Diffusion_FFPE(model_path=args.model_path, lora_rank_unet=args.lora_rank_unet,
                               lora_rank_vae=args.lora_rank_vae, gradient_checkpointing=args.gradient_checkpointing,
                               enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention)
    model_b2a = Diffusion_FFPE(model_path=args.model_path, lora_rank_unet=args.lora_rank_unet,
                               lora_rank_vae=args.lora_rank_vae, gradient_checkpointing=args.gradient_checkpointing,
                               enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention)

    net_disc_a = initialize_discriminator(args.gan_loss_type)
    net_disc_b = initialize_discriminator(args.gan_loss_type)

    model_a2b.to(accelerator.device, dtype=weight_dtype)
    model_b2a.to(accelerator.device, dtype=weight_dtype)
    net_disc_a.to(accelerator.device, dtype=weight_dtype)
    net_disc_b.to(accelerator.device, dtype=weight_dtype)

    # Define Optimizer and Loss
    params_gen = list(model_a2b.get_trainable_params()) + list(model_b2a.get_trainable_params())
    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler_gen = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )
    lr_scheduler_disc = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )
    
    crit_bridge = torch.nn.MSELoss()
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # Bridge schedule
    bridge_alphas = get_bridge_schedule(args.num_bridge_steps, args.bridge_schedule).to(accelerator.device)

    # Start Training Loop
    epoch = 0
    global_step = 0
    if args.resume:
        resume = torch.load(os.path.join(args.ckpt_path, "resume_latest.pkl"))
        epoch = resume['epoch']
        global_step = resume['global_step']
        net_disc_a.load_state_dict(resume['net_disc_a'], strict=True)
        net_disc_b.load_state_dict(resume['net_disc_b'], strict=True)
        optimizer_gen.load_state_dict(resume['optimizer_gen'])
        optimizer_disc.load_state_dict(resume['optimizer_disc'])
        lr_scheduler_gen.load_state_dict(resume['lr_scheduler_gen'])
        lr_scheduler_disc.load_state_dict(resume['lr_scheduler_disc'])
        
        # Load both bridge models
        sd_a2b = torch.load(os.path.join(args.ckpt_path, f"model_a2b_{global_step}.pkl"))
        sd_b2a = torch.load(os.path.join(args.ckpt_path, f"model_b2a_{global_step}.pkl"))
        model_a2b.load_ckpt_from_state_dict(sd_a2b)
        model_b2a.load_ckpt_from_state_dict(sd_b2a)

    model_a2b, model_b2a, net_disc_a, net_disc_b = accelerator.prepare(
        [model_a2b, model_b2a, net_disc_a, net_disc_b]
    )
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.\
        prepare([net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc])

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        if not os.path.exists(ref_path):
            feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
            features, mu, sigma = get_mu_sigma(args.valid_target_folder, feat_model)
            np.savez(ref_path, mu=mu, sigma=sigma, features=features)
            del feat_model
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    for epoch in range(epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model_a2b):
                img_a = batch["pixel_values_src"].to(accelerator.device, dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(accelerator.device, dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)

                """
                Diffusion Bridge Objective A->B
                Sample random timestep and create bridge interpolation
                """
                # Sample random timestep t from [1, num_bridge_steps-1]
                t_idx = torch.randint(1, args.num_bridge_steps, (1,), device=accelerator.device).item()
                alpha_t = bridge_alphas[t_idx].view(1, 1, 1, 1)
                alpha_prev = bridge_alphas[t_idx - 1].view(1, 1, 1, 1)
                
                # Create interpolated state x_t between img_a and img_b
                x_t_a2b = bridge_interpolate(img_a, img_b, alpha_t)
                
                # Model predicts the direction toward img_b
                pred_b = model_a2b(x_t_a2b, "a2b", fixed_a2b_emb)
                
                # Target: the next step in the bridge (closer to img_b)
                with torch.no_grad():
                    x_target_a2b = bridge_interpolate(img_a, img_b, alpha_prev)
                
                # Bridge matching loss: predicted should match the bridge trajectory
                loss_bridge_a2b = crit_bridge(pred_b, img_b) * args.lambda_bridge
                loss_bridge_a2b += net_lpips(pred_b, img_b).mean() * args.lambda_bridge_lpips

                """
                Diffusion Bridge Objective B->A
                """
                x_t_b2a = bridge_interpolate(img_b, img_a, alpha_t)
                pred_a = model_b2a(x_t_b2a, "b2a", fixed_b2a_emb)
                
                with torch.no_grad():
                    x_target_b2a = bridge_interpolate(img_b, img_a, alpha_prev)
                
                loss_bridge_b2a = crit_bridge(pred_a, img_a) * args.lambda_bridge
                loss_bridge_b2a += net_lpips(pred_a, img_a).mean() * args.lambda_bridge_lpips

                # Backward for bridge losses
                accelerator.backward(loss_bridge_a2b + loss_bridge_b2a, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) - Generate full translations
                """
                fake_b = model_a2b(img_a, "a2b", fixed_a2b_emb)
                fake_a = model_b2a(img_b, "b2a", fixed_b2a_emb)
                
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Optimal Transport Consistency Loss
                Enforce that A->B->A and B->A->B preserve content
                """
                with torch.no_grad():
                    # For consistency, we want the bridge to be reversible
                    recon_a = model_b2a(fake_b, "b2a", fixed_b2a_emb)
                    recon_b = model_a2b(fake_a, "a2b", fixed_a2b_emb)
                
                loss_ot_a = crit_bridge(recon_a, img_a) * args.lambda_ot
                loss_ot_b = crit_bridge(recon_b, img_b) * args.lambda_ot
                
                accelerator.backward(loss_ot_a + loss_ot_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for A->B (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_D_A_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc_a.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for A->B (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_D_A_real, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc_a.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for B->A (fake inputs)
                """
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_D_B_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc_b.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for B->A (real inputs)
                """
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_D_B_real, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc_b.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {
                "bridge_a2b": loss_bridge_a2b.detach().item(), 
                "bridge_b2a": loss_bridge_b2a.detach().item(),
                "gan_a": loss_gan_a.detach().item(), 
                "gan_b": loss_gan_b.detach().item(),
                "disc_a": (loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()),
                "disc_b": (loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()),
                "ot_a": loss_ot_a.detach().item(),
                "ot_b": loss_ot_b.detach().item(),
            }

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_model_a2b = accelerator.unwrap_model(model_a2b)
                    eval_model_b2a = accelerator.unwrap_model(model_b2a)
                    eval_disc_a = accelerator.unwrap_model(net_disc_a)
                    eval_disc_b = accelerator.unwrap_model(net_disc_b)

                    if global_step % args.viz_freq == 0:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(),
                                                                 caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(),
                                                                 caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/fake_b": [wandb.Image(fake_b[idx].float().detach().cpu(),
                                                                 caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/fake_a": [wandb.Image(fake_a[idx].float().detach().cpu(),
                                                                 caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/recon_a": [wandb.Image(recon_a[idx].float().detach().cpu(),
                                                                  caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/recon_b": [wandb.Image(recon_b[idx].float().detach().cpu(),
                                                                  caption=f"idx={idx}") for idx in range(bsz)]}
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 0:
                        # Save model A->B
                        out_path_a2b = os.path.join(args.output_dir, "checkpoints", f"model_a2b_{global_step}.pkl")
                        sd_a2b = {
                            "rank_unet": args.lora_rank_unet,
                            "unet_conv_in": eval_model_a2b.unet.conv_in.state_dict(),
                            "sd_encoder": get_peft_model_state_dict(eval_model_a2b.unet, adapter_name="default_encoder"),
                            "sd_decoder": get_peft_model_state_dict(eval_model_a2b.unet, adapter_name="default_decoder"),
                            "sd_other": get_peft_model_state_dict(eval_model_a2b.unet, adapter_name="default_others"),
                            "rank_vae": args.lora_rank_vae,
                            "sd_vae_enc": eval_model_a2b.vae_enc.state_dict(),
                            "sd_vae_dec": eval_model_a2b.vae_dec.state_dict(),
                        }
                        torch.save(sd_a2b, out_path_a2b)
                        
                        # Save model B->A
                        out_path_b2a = os.path.join(args.output_dir, "checkpoints", f"model_b2a_{global_step}.pkl")
                        sd_b2a = {
                            "rank_unet": args.lora_rank_unet,
                            "unet_conv_in": eval_model_b2a.unet.conv_in.state_dict(),
                            "sd_encoder": get_peft_model_state_dict(eval_model_b2a.unet, adapter_name="default_encoder"),
                            "sd_decoder": get_peft_model_state_dict(eval_model_b2a.unet, adapter_name="default_decoder"),
                            "sd_other": get_peft_model_state_dict(eval_model_b2a.unet, adapter_name="default_others"),
                            "rank_vae": args.lora_rank_vae,
                            "sd_vae_enc": eval_model_b2a.vae_enc.state_dict(),
                            "sd_vae_dec": eval_model_b2a.vae_dec.state_dict(),
                        }
                        torch.save(sd_b2a, out_path_b2a)
                        
                        # Save resume state
                        resume_path = os.path.join(args.output_dir, "checkpoints", f"resume_latest.pkl")
                        resume = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'net_disc_a': eval_disc_a.state_dict(),
                            'net_disc_b': eval_disc_b.state_dict(),
                            'optimizer_gen': optimizer_gen.state_dict(),
                            'optimizer_disc': optimizer_disc.state_dict(),
                            'lr_scheduler_gen': lr_scheduler_gen.state_dict(),
                            'lr_scheduler_disc': lr_scheduler_disc.state_dict(),
                        }
                        torch.save(resume, resume_path)
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 0:
                        net_dino = DinoStructureLoss()
                        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
                        """
                        Evaluate "A->B"
                        """
                        fid_dir = os.path.join(args.output_dir, f"fid-{global_step}")
                        fid_output_dir = os.path.join(fid_dir, "samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = evaluate(eval_model_a2b, net_dino, l_images_src_test, fixed_a2b_emb, 'a2b',
                                                     fid_output_dir, args.val_img_prep, args.validation_num_images)
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        score_fid_a2b = calculate_fid(ref_path, fid_output_dir, feat_model)
                        print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")
                        logs["val/fid_a2b"] = score_fid_a2b
                        logs["val/dino_struct_a2b"] = dino_score_a2b
                        shutil.rmtree(fid_dir)
                        del net_dino, feat_model  # free up memory

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == '__main__':
    training_args = parse_args_training()
    main(training_args)