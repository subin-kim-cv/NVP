'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import numpy as np
import os

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, summary_fn):

    optim = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-5)


    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    tmp_psnr = 0
    best_psnr = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []


        for epoch in range(epochs):
            
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            
            for step, (model_input, gt) in enumerate(train_dataloader):
            
                # GPU
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                gt["img"] = gt["img"].float()
                gt["img"] = (gt["img"]-127.5)/(127.5)

                model_output = model(model_input)

                losses = loss_fn(model_output, gt)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    tmp_psnr = 10*torch.log10(4/single_loss)
                    writer.add_scalar(loss_name+"_psnr", tmp_psnr, total_steps)
                    writer.add_scalar("lr", float(scheduler.get_last_lr()[0]), total_steps)
                    train_loss += single_loss


                if tmp_psnr > best_psnr and not (total_steps+1) % (200):
                    torch.save({'epoch': total_steps,
                                        'model': model.state_dict(),
                                        'optimizer': optim.state_dict(),
                                        'scheduler': scheduler.state_dict(),    
                                        }, os.path.join(checkpoints_dir, 'model_best.pth'))

                    best_psnr = tmp_psnr

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                model_output = None
                
                if not total_steps % steps_til_summary:
                    psnr = summary_fn(model, model_input, gt, writer, total_steps)
                    tqdm.write("Epoch %d, Total loss %0.6f, psnr: %0.6f" % (epoch, train_loss, psnr))

                pbar.update(1)
                total_steps += 1

        torch.save({'epoch': total_steps,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),    
                    }, os.path.join(checkpoints_dir, f'model_final.pth'))
                                                
        psnr = summary_fn(model, model_input, gt, writer, total_steps)
        writer.close()
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))   

        return psnr

