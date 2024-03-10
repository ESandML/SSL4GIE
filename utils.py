from Models import models


def get_BarlowTwins_backbone(weight_path, head, num_classes, frozen, dense):
    return models.ResNet_from_Any(weight_path, head, num_classes, frozen, dense)


def get_MAE_backbone(
    weight_path, head, num_classes, frozen, dense, det, fixed_size=None, out_token="cls"
):
    return models.ViT_from_MAE(
        weight_path,
        head,
        num_classes,
        frozen,
        dense,
        det,
        fixed_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_token=out_token,
    )


def get_MoCoV3_backbone(
    weight_path,
    arch,
    head,
    num_classes,
    frozen,
    dense,
    det,
    fixed_size=None,
    out_token="cls",
):
    if arch == "vit_b":
        return models.ViT_from_MoCoV3(
            weight_path,
            head,
            num_classes,
            frozen,
            dense,
            det,
            fixed_size,
            embed_dim=768,
            out_token=out_token,
        )
    elif arch == "resnet50":
        return models.ResNet_from_Any(weight_path, head, num_classes, frozen, dense)


def get_ImageNet_or_random_ResNet(head, num_classes, frozen, dense, ImageNet_weights):
    return models.ResNet_from_Any(
        None, head, num_classes, frozen, dense, ImageNet_weights
    )


def get_ImageNet_or_random_ViT(
    head,
    num_classes,
    frozen,
    dense,
    det,
    ImageNet_weights,
    fixed_size=None,
    out_token="cls",
):
    return models.VisionTransformer_from_Any(
        head,
        num_classes,
        frozen,
        dense,
        det,
        fixed_size,
        768,
        12,
        12,
        out_token,
        ImageNet_weights,
    )

