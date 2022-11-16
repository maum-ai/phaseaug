# modified from https://github.com/jik876/hifi-gan
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import (
    Generator, 
    MultiPeriodDiscriminator, 
    MultiScaleDiscriminator, 
    feature_loss, 
    generator_loss, 
    discriminator_loss
)
from utils import (
    plot_spectrogram, 
    scan_checkpoint, 
    load_checkpoint, 
    save_checkpoint
)
from phaseaug import PhaseAug
from math import pi

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'],
                           init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus,
                           rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    if a.aug:
        aug = PhaseAug(
            a.aug_nfft,
            a.aug_hop,
            a.filter,
            a.var,
            a.delta_max,
            a.cutoff,
            a.half_width,
            a.kernel_size,
            a.padding).to(device)
        phi_ref = torch.arange(
            a.aug_nfft // 2 + 1,
            device=device).unsqueeze(0) * 2 * pi / (a.aug_nfft) 

    periods = ['2', '3', '5', '7', '11', 'all']
    scales = ['1', '2', '4', 'all']

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path) and a.resume:
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    else:
        cp_g = None
        cp_do = None

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator,
                                            device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(),
                                h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(),
                                                mpd.parameters()),
                                h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g,
                                                         gamma=h.lr_decay,
                                                         last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d,
                                                         gamma=h.lr_decay,
                                                         last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist,
                          h.segment_size,
                          h.n_fft,
                          h.num_mels,
                          h.hop_size,
                          h.win_size,
                          h.sampling_rate,
                          h.fmin,
                          h.fmax,
                          n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True,
                          fmax_loss=h.fmax_for_loss,
                          device=device,
                          fine_tuning=a.fine_tuning,
                          base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset,
                              num_workers=h.num_workers,
                              shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist,
                              h.segment_size,
                              h.n_fft,
                              h.num_mels,
                              h.hop_size,
                              h.win_size,
                              h.sampling_rate,
                              h.fmin,
                              h.fmax,
                              False,
                              False,
                              n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss,
                              device=device,
                              fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset,
                                       num_workers=1,
                                       shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, f'logs/{a.name}'))

    generator.train()
    mpd.train()
    msd.train()
    if a.aug:
        aug.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            B = x.shape[0]
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device,
                                                     non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft,
                                          h.num_mels, h.sampling_rate,
                                          h.hop_size, h.win_size, h.fmin,
                                          h.fmax_for_loss)

            optim_d.zero_grad()

            if a.aug:
                mu = aug.sample_mu(B, device)
                phi = mu * phi_ref
                aug_y = aug(y, phi)
                aug_y_g = aug(y_g_hat, phi).detach()

            # MPD
            if a.aug and (not a.aug_msd_only):
                y_df_hat_r, y_df_hat_g, _, _ = mpd(aug_y, aug_y_g)
            else:
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g, \
                accs_f_r, accs_f_g, dfr_stats, dfg_stats = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

            # MSD
            if a.aug:
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(aug_y, aug_y_g)
            else:
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g, \
                accs_s_r, accs_s_g, dsr_stats, dsg_stats = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )
            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            if a.aug:
                mu = aug.sample_mu(B, device)
                phi = mu * phi_ref
                aug_y = aug(y, phi)
                aug_y_g = aug(y_g_hat, phi)
                if not a.aug_msd_only:
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        aug_y, aug_y_g)
                else:
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                    aug_y, aug_y_g)
            else:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'
                        .format(steps, loss_gen_all, mel_error,
                                time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(
                        a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': (generator.module if h.num_gpus > 1
                                          else generator).state_dict()
                        })
                    checkpoint_path = "{}/do_{:08d}".format(
                        a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': (mpd.module
                                    if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module
                                    if h.num_gpus > 1 else msd).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        })

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/gen_disc_s", loss_gen_s, steps)
                    sw.add_scalar("training/gen_disc_f", loss_gen_f, steps)
                    sw.add_scalar("training/gen_fm_s", loss_fm_s, steps)
                    sw.add_scalar("training/gen_fm_f", loss_fm_f, steps)
                    sw.add_scalar("training/gen_mel", loss_mel, steps)
                    sw.add_scalar("training/disc_s", loss_disc_s, steps)
                    sw.add_scalar("training/disc_f", loss_disc_f, steps)
                    sw.add_scalar("training/disc_total", loss_disc_all, steps)
                    for period_index, period in enumerate(periods):
                        sw.add_scalar(f"training/acc_f_r_{period}",
                                      accs_f_r[period_index], steps)
                        sw.add_scalar(f"training/acc_f_g_{period}",
                                      accs_f_g[period_index], steps)
                        sw.add_scalar(f"training/d_f_r_{period}_mean",
                                      dfr_stats[period_index][0], steps)
                        sw.add_scalar(f"training/d_f_r_{period}_std",
                                      dfr_stats[period_index][1], steps)
                        sw.add_scalar(f"training/d_f_g_{period}_mean",
                                      dfg_stats[period_index][0], steps)
                        sw.add_scalar(f"training/d_f_g_{period}_std",
                                      dfg_stats[period_index][1], steps)

                    for scale_index, scale in enumerate(scales):
                        sw.add_scalar(f"training/acc_s_r_{scale}",
                                      accs_s_r[scale_index], steps)
                        sw.add_scalar(f"training/acc_s_g_{scale}",
                                      accs_s_g[scale_index], steps)
                        sw.add_scalar(f"training/d_s_r_{scale}_mean",
                                      dsr_stats[scale_index][0], steps)
                        sw.add_scalar(f"training/d_s_r_{scale}_std",
                                      dsr_stats[scale_index][1], steps)
                        sw.add_scalar(f"training/d_s_g_{scale}_mean",
                                      dsg_stats[scale_index][0], steps)
                        sw.add_scalar(f"training/d_s_g_{scale}_std",
                                      dsg_stats[scale_index][1], steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    mpd.eval()
                    msd.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    val_fm_s_tot = 0.
                    val_fm_f_tot = 0.
                    val_disc_s_tot = 0.
                    val_disc_f_tot = 0.
                    val_accs_s_r_tot = [0.] * len(scales)
                    val_accs_s_g_tot = [0.] * len(scales)
                    val_accs_f_r_tot = [0.] * len(periods)
                    val_accs_f_g_tot = [0.] * len(periods)
                    val_accs_s_r = [0.] * len(scales)
                    val_accs_s_g = [0.] * len(scales)
                    val_accs_f_r = [0.] * len(periods)
                    val_accs_f_g = [0.] * len(periods)
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(
                                y_mel.to(device, non_blocking=True))
                            y = y[..., :y_g_hat.shape[-1]].unsqueeze(1).to(
                                device)
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size,
                                h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                            # MPD
                            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                                y, y_g_hat.detach())
                            loss_disc_f, losses_disc_f_r, losses_disc_f_g, \
                                accs_f_r, accs_f_g, dfr_stats, dfg_stats = discriminator_loss(
                                    y_df_hat_r, y_df_hat_g
                                )

                            # MSD
                            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                                y, y_g_hat.detach())
                            loss_disc_s, losses_disc_s_r, losses_disc_s_g, \
                                accs_s_r, accs_s_g, dsr_stats, dsg_stats = discriminator_loss(
                                    y_ds_hat_r, y_ds_hat_g
                                )
                            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                            val_fm_f_tot += loss_fm_f
                            val_fm_s_tot += loss_fm_s
                            val_disc_f_tot += loss_disc_f
                            val_disc_s_tot += loss_disc_s
                            for period_index, period in enumerate(periods):
                                val_accs_f_r_tot[period_index] += accs_f_r[
                                    period_index]
                                val_accs_f_g_tot[period_index] += accs_f_g[
                                    period_index]
                            for scale_index, scale in enumerate(scales):
                                val_accs_s_r_tot[scale_index] += accs_s_r[
                                    scale_index]
                                val_accs_s_g_tot[scale_index] += accs_s_g[
                                    scale_index]

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0],
                                                 steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j),
                                                  plot_spectrogram(x[0]),
                                                  steps)

                                sw.add_audio('generated/y_hat_{}'.format(j),
                                             y_g_hat[0], steps,
                                             h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                    h.sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax)
                                sw.add_figure(
                                    'generated/y_hat_spec_{}'.format(j),
                                    plot_spectrogram(
                                        y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps)

                        val_err = val_err_tot / (j + 1)
                        val_fm_f = val_fm_f_tot / (j + 1)
                        val_fm_s = val_fm_s_tot / (j + 1)
                        val_disc_f = val_disc_f_tot / (j + 1)
                        val_disc_s = val_disc_s_tot / (j + 1)
                        for period_index, period in enumerate(periods):
                            val_accs_f_r[period_index] = val_accs_f_r_tot[
                                period_index] / (j + 1)
                            val_accs_f_g[period_index] = val_accs_f_g_tot[
                                period_index] / (j + 1)
                        for scale_index, scale in enumerate(scales):
                            val_accs_s_r[scale_index] = val_accs_s_r_tot[
                                scale_index] / (j + 1)
                            val_accs_s_g[scale_index] = val_accs_s_g_tot[
                                scale_index] / (j + 1)

                        sw.add_scalar("validation/mel_spec_error", val_err,
                                      steps)
                        sw.add_scalar("validation/fm_s", val_fm_s, steps)
                        sw.add_scalar("validation/fm_f", val_fm_f, steps)
                        sw.add_scalar("validation/disc_s", val_disc_s, steps)
                        sw.add_scalar("validation/disc_f", val_disc_f, steps)
                        for period_index, period in enumerate(periods):
                            sw.add_scalar(f"validation/acc_f_r_{period}",
                                          val_accs_f_r[period_index], steps)
                            sw.add_scalar(f"validation/acc_f_g_{period}",
                                          val_accs_f_g[period_index], steps)
                        for scale_index, scale in enumerate(scales):
                            sw.add_scalar(f"validation/acc_s_r_{scale}",
                                          val_accs_s_r[scale_index], steps)
                            sw.add_scalar(f"validation/acc_s_g_{scale}",
                                          val_accs_s_g[scale_index], steps)

                    generator.train()
                    mpd.train()
                    msd.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, required=True)
    parser.add_argument('--input_wavs_dir',
                        default='path/LJSpeech-1.1/wavs_22k')
    parser.add_argument('--input_mels_dir',
                        default='path/LJSpeech-1.1/wavs_22k')
    parser.add_argument('--input_training_file',
                        default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file',
                        default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=5000, type=int)
    parser.add_argument('--stdout_interval', default=10, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=50000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    ### PhaseAug related args
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--aug_nfft', default=1024, type=int)
    parser.add_argument('--aug_hop', default=256, type=int)
    parser.add_argument('--aug_msd_only', action='store_true')  # not effective
    parser.add_argument('--data_ratio', default=1., type=float)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--var', default=6., type=float)
    parser.add_argument('--delta_max', default=2., type=float)
    ### Low Pass Filter related args
    parser.add_argument('--cutoff', default=0.05, type=float)
    parser.add_argument('--half_width', default=0.012, type=float)  # Paper will be modified in rebuttal phase.
    parser.add_argument('--kernel_size', default=128, type=int)
    parser.add_argument('--padding', default='constant', type=str)
    ###
    parser.add_argument('--resume', action='store_true')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    if a.data_ratio < 1.:
        a.training_epochs = a.training_epochs * int(1. / a.data_ratio)
        h.lr_decay = h.lr_decay**a.data_ratio

    a.checkpoint_path = os.path.join(a.checkpoint_path, a.name)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(
            a,
            h,
        ))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
