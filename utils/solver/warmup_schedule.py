"""
该文件构建warmup预热机制
"""
# Build warmup scheduler


def build_warmup(d_cfg, base_lr=0.01):
    print('==============================')
    print('WarmUpScheduler: {}'.format(d_cfg['warmup']))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_factor: {}'.format(d_cfg['warmup_factor']))
    print('--wp_iter: {}'.format(d_cfg['wp_iter']))

    warmup_scheduler = WarmUpScheduler(
        name=d_cfg['warmup'],
        base_lr=base_lr, 
        wp_iter=d_cfg['wp_iter'],
        warmup_factor=d_cfg['warmup_factor']
        )
    
    return warmup_scheduler

                           
# Basic Warmup Scheduler
class WarmUpScheduler(object):
    def __init__(self, 
                 name='linear', 
                 base_lr=0.01, 
                 wp_iter=500, 
                 warmup_factor=0.00066667):
        self.name = name
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor

    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']  # MultiStepLR的初始学习率
            ratio = init_lr / base_lr  # base_lr只作为一个基本单位，不影响当前学习率，最后lr=init_lr * warmup_factor
            param_group['lr'] = lr * ratio

    def warmup(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.wp_iter, 4)
            self.set_lr(optimizer, tmp_lr, self.base_lr)

        elif self.name == 'linear':
            alpha = iter / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # 从初始值self.warmup_factor逐渐变成1
            tmp_lr = self.base_lr * warmup_factor  # base_lr作为一个基本单位
            self.set_lr(optimizer, tmp_lr, self.base_lr)

    def __call__(self, iter, optimizer):
        self.warmup(iter, optimizer)
        