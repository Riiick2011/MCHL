from .yowo.build import build_yowo


def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):  # 该函数用于构建一个YOWO检测器，并根据是否处于训练中返回损失函数
    # build action detector
    if 'yowo_' in args.version:
        model, criterion = build_yowo(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
            )

    return model, criterion

