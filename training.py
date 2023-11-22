import torch
import utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from loss_functions import compute_psnr
import torch.nn as nn

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 4000, 6000, 8000], gamma=0.5)

    for param_group in optim.param_groups:
        for param in param_group['params']:
            if param.requires_grad:
                print(param.name, param.data.size())

    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    best_psnr = 0

    with tqdm(total=epochs) as pbar:
        train_losses = []

        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save({"model": model.state_dict(),
                            "latent_grid": model.latent_grid},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            
            for step, (model_input, gt) in enumerate(train_dataloader):
            
                # GPU
                model_input = model_input.float().cuda()
                model_input = model_input

                gt_coords = gt[0].float().cuda()
                gt_values = gt[1].float().cuda().squeeze()
                model_output = model(gt_coords, model_input)

                loss = loss_fn(model_output['model_out'], gt_values)
                train_loss.add(loss.item())

                psnr = compute_psnr(model_output['model_out'], gt_values)

                if psnr > best_psnr:
                    torch.save({'epoch': total_steps,
                                        'model': model.state_dict(),
                                        'optimizer': optim.state_dict(),
                                        'scheduler': scheduler.state_dict(),
                                        'latent_grid': model.latent_grid,    
                                        }, os.path.join(checkpoints_dir, 'model_best.pth'))

                    best_psnr = psnr

                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                if not total_steps % steps_til_summary:
                     # total_loss = round(sum(train_loss) / len(train_loss), 5)
                     tqdm.write("Epoch {}, Total loss {}, PSNR {}".format(epoch, train_loss.item(), psnr))

                pbar.update(1)
                total_steps += 1

        torch.save({'epoch': total_steps,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),  
                    'latent_grid': model.latent_grid,  
                    }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
        return psnr

