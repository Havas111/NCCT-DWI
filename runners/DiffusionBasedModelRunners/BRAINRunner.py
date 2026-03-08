import os
import torch.optim.lr_scheduler
import time
from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm


@Registers.runners.register_with_name('BRAINRunner')
class BRAINRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BRAIN":
            brainnet = BrownianBridgeModel(config.model)
        else:
            raise NotImplementedError
        brainnet.apply(weights_init)
        return brainnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                if self.config.training.use_DDP:
                    device = torch.device(f"cuda:{self.config.training.local_rank}")
                else:
                    device = self.config.training.device[0]

                self.net.ori_latent_mean = states['ori_latent_mean'].to(device)
                self.net.ori_latent_std = states['ori_latent_std'].to(device)
                self.net.cond_latent_mean = states['cond_latent_mean'].to(device)
                self.net.cond_latent_std = states['cond_latent_std'].to(device)
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        self.logger("Total Number of parameter: %.2fM" % (total_num / 1e6))
        self.logger("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        if hasattr(scheduler, 'verbose'):
            scheduler.verbose = True
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states


    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        return 0

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        x = batch['MR']
        x_cond = batch['CT']

        if self.config.training.use_DDP:
            device = torch.device(f"cuda:{self.config.training.local_rank}")
        else:
            device = self.config.training.device[0]
        batch_size = x.shape[0] if x.shape[0] < 8 else 8
        x = x[0:batch_size].to(device)
        x_cond = x_cond[0:batch_size].to(device)

        grid_size = 8
        sample = net.sample(x_cond, context=None, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        x_cond_vis = x_cond[:, 0:1, :, :].to('cpu')

        image_grid = get_image_grid(x_cond_vis.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path,):
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num

        total_start = time.time()
        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01, desc="Sampling")
        resume_mode = True
        if resume_mode:
            print(f"Resume Mode")
        else:
            print(f"Restart Mode")

        for batch_idx, test_batch in enumerate(pbar):

            subject_names = test_batch['subject']
            if resume_mode:
                batch_already_processed = True
                for name in subject_names:
                    if sample_num > 1:
                        check_path = os.path.join(result_path, name, 'output_0.png')
                    else:
                        check_path = os.path.join(result_path, f'{name}.png')
                    if not os.path.exists(check_path):
                        batch_already_processed = False
                        break
                if batch_already_processed:
                    continue

            batch_start = time.time()

            x = test_batch['MR']
            x_cond = test_batch['CT']

            if self.config.training.use_DDP:
                device = torch.device(f"cuda:{self.config.training.local_rank}")
            else:
                device = self.config.training.device[0]
            subject_names = test_batch['subject']
            x = x.to(device)
            x_cond = x_cond.to(device)

            for j in range(sample_num):
                sample = net.sample(x_cond, context=None, clip_denoised=False)

                for i in range(batch_size):
                    condition = x_cond[i, 0:1, :, :].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    file_prefix = subject_names[i]

                    if j == 0:

                        save_single_image(condition, condition_path, f'{file_prefix}_CT.png', to_normal=to_normal)
                        save_single_image(gt, gt_path, f'{file_prefix}_MR.png', to_normal=to_normal)

                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, file_prefix))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{file_prefix}.png', to_normal=to_normal)


            batch_time = time.time() - batch_start
            pbar.set_postfix({'Batch Time (s)': f'{batch_time:.2f}'})

        total_time = time.time() - total_start
        print(f"\n✅ All inference done in {total_time:.2f} seconds. "
              f"Average per batch: {total_time / len(test_loader):.2f} seconds.")
