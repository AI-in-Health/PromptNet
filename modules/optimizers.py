import torch
from torch import optim


def build_optimizer(args, model):
    if args.train_mode == 'full' and args.clip_update == 'yes':
        # If F_version is v1, include all parameters of align_model
        if args.F_version == 'v1':
            align_model_params = model.align_model.parameters()
        # If F_version is v2, include only vision parameters of align_model
        elif args.F_version == 'v2':
            align_model_params = model.align_model.vision_model.parameters()
        else:
            raise ValueError(f"Unknown F_version: {args.F_version}")

        ve_params = list(map(id, align_model_params))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
            [{'params': align_model_params, 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            betas=args.adam_betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        optimizer = getattr(torch.optim, args.optim)(
            model.parameters(),
            lr=args.lr_ve,
            betas=args.adam_betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler

