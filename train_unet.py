from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
import torch
import torch.nn.functional as F
import time
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchdistpackage import setup_distributed_slurm
from actnn.ops import quantize_activation, dequantize_activation
from actnn import config
# def quantize_activation(x, bits=4):
#     import math
#     # get range
#     min_n = torch.min(x).item()
#     max_n = torch.max(x).item()
#     # import pdb;pdb.set_trace()
#     range = max(abs(max_n), abs(min_n))
#     scale = range/(math.pow(2,(bits-1)) -1)
#     scaled_x = x/scale
#     # clip to ints
#     ints = torch.round(scaled_x)
#     # make ints into dense int32

#     return ints, scale, bits

# def dequantize_activation(infos):
#     ints, scale, bits = infos
#     unscaled = ints*scale
#     return unscaled

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def batch_inp(bs20inp, target_bs):
    mbs=2
    bs4inp = bs20inp[:mbs]
    if target_bs<mbs:
        return bs20inp[:target_bs]
    if target_bs==mbs:
        return bs20inp
    num = int(target_bs/mbs)
    out = torch.cat([bs4inp.clone().detach() for _ in  range(num)])
    if out.dim()==4:
        out = out.to(memory_format=torch.channels_last).contiguous()
    print(out.shape)
    return out



def train(model, vae, optimizer_class, batchsize, use_zero=False, use_amp=True, h=512, w=512, is_xl=False,
          quant_act=True):
    if not is_xl:
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        prompt_embeds = torch.rand([batchsize,77,768], dtype=torch.float16).cuda()
        time_ids = torch.rand([batchsize,6], dtype=torch.float16).cuda()
        text_embeds = torch.rand([batchsize,1280], dtype=torch.float16).cuda()
        encoder_hidden_states = torch.rand([batchsize,77,768], dtype=torch.float32).cuda()

    else:
        dt=torch.float32
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        prompt_embeds = torch.rand([batchsize,77,2048], dtype=dt).cuda()
        time_ids = torch.rand([batchsize,6], dtype=dt).cuda()
        text_embeds = torch.rand([batchsize,1280], dtype=dt).cuda()


    model_input = torch.rand([batchsize, 3, h, w], dtype=torch.float32).cuda()

    unet_added_conditions = {
        "time_ids": time_ids,
        "text_embeds": text_embeds
    }


    model.enable_gradient_checkpointing()
    # model.enable_xformers_memory_efficient_attention()
    # torch._dynamo.config.suppress_errors = True
    # model=torch.compile(model)
    perf_times = []
    if use_zero:
        model = FSDP(model, sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP)
        opt =optimizer_class(model.parameters())
    else:
        opt =optimizer_class(model.parameters())

    def pack(x):
        sp = x.shape
        quantized = quantize_activation(x, None)

        return (quantized, sp)

    def unpack(args):
        # import pdb;pdb.set_trace()
        quantized, shape = args
        x = dequantize_activation(quantized, shape)
        return x


    from torch.profiler import profile, record_function, ProfilerActivity
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=0, warmup=5, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof/unet_720p_tp2'),
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True)
    # prof.start()
    for ind in range(20):
        torch.cuda.synchronize()
        vae_beg = time.time()
        with torch.no_grad():
            noisy_model_input = vae.encode(model_input).latent_dist.sample().mul_(0.18215)
        torch.cuda.synchronize()

        beg = time.time()

        with torch.autocast(dtype=torch.float16, device_type='cuda', enabled=use_amp), torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        # with torch.autocast(dtype=torch.float16, device_type='cuda', enabled=use_amp):
            if is_xl:
                model_pred = model(
                            noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                        ).sample
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
            else:
                model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")

        loss.backward()

        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        if ind>10:
           perf_times.append(time.time()-beg)
        beg=time.time()
    print("max mem", torch.cuda.max_memory_allocated()/1e9)
    print(perf_times)

enable_tf32()
rank, world_size, port, addr=setup_distributed_slurm()

pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"

# pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=None,
    low_cpu_mem_usage=False, device_map=None
).cuda()
unet.train()
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()
optimizer_class = functools.partial(torch.optim.Adam,fused = True)
train(unet, vae, optimizer_class, 4,
      use_amp=False, use_zero=False, h=576, w=1080, is_xl ='xl' in pretrained_model_name_or_path,
      quant_act=True)