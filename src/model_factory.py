import torch
import torch.nn as nn

from torchvision import models

MODEL_REGISTRY = {
    "mobilenet_v3_large": {
        "constructor": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "family": "CNN"
    },

    "efficientnet_b0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "family": "CNN"
    },

    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V2,
        "family": "CNN"
    },

    "vit_b_16": {
        "constructor": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "family": "Transformer"
    },

    "swin_t": {
        "constructor": models.swin_t,
        "weights": models.Swin_T_Weights.IMAGENET1K_V1,
        "family": "Transformer"
    }
}


def get_available_models():

    return list(MODEL_REGISTRY.keys())


def replace_classifier(
    model,
    model_name,
    num_classes,
    dropout=0.2
):

    if model_name == "mobilenet_v3_large":

        in_features = model.classifier[-1].in_features

        model.classifier[-2] = nn.Dropout(p=dropout)

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "efficientnet_b0":

        in_features = model.classifier[-1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "vit_b_16":

        in_features = model.heads.head.in_features

        model.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "swin_t":

        in_features = model.head.in_features

        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    else:

        raise ValueError(
            f"Arquitectura no soportada: {model_name}"
        )

    return model


def get_classifier(model, model_name):

    if model_name in [
        "mobilenet_v3_large",
        "efficientnet_b0"
    ]:

        return model.classifier

    elif model_name == "resnet50":

        return model.fc

    elif model_name == "vit_b_16":

        return model.heads

    elif model_name == "swin_t":

        return model.head

    raise ValueError(
        f"Arquitectura no soportada: {model_name}"
    )


def get_finetuning_modules(model, model_name):

    if model_name == "mobilenet_v3_large":

        return [
            model.features[-2:]
        ]

    elif model_name == "efficientnet_b0":

        return [
            model.features[-2:]
        ]

    elif model_name == "resnet50":

        return [
            model.layer4
        ]

    elif model_name == "vit_b_16":

        return [
            model.encoder.layers[-2:],
            model.encoder.ln
        ]

    elif model_name == "swin_t":

        return [
            model.features[-1],
            model.norm
        ]

    raise ValueError(
        f"Arquitectura no soportada: {model_name}"
    )


def configure_trainable_layers(
    model,
    model_name,
    mode="feature_extraction"
):

    valid_modes = [
        "feature_extraction",
        "partial_finetuning",
        "full_finetuning"
    ]

    if mode not in valid_modes:

        raise ValueError(
            f"Modalidad no válida: {mode}"
        )

    # Congelar inicialmente todos los parámetros
    for param in model.parameters():

        param.requires_grad = False

    if mode == "full_finetuning":

        for param in model.parameters():

            param.requires_grad = True

    else:

        # Descongelar el clasificador
        classifier = get_classifier(
            model,
            model_name
        )

        for param in classifier.parameters():

            param.requires_grad = True

        if mode == "partial_finetuning":

            modules = get_finetuning_modules(
                model,
                model_name
            )

            for module in modules:

                for param in module.parameters():

                    param.requires_grad = True

    model.training_mode_name = mode

    return model


def set_model_train_mode(
    model,
    model_name
):

    mode = model.training_mode_name

    if mode == "full_finetuning":

        model.train()

    else:

        # Mantener el backbone en evaluación
        model.eval()

        # Activar entrenamiento del clasificador
        classifier = get_classifier(
            model,
            model_name
        )

        classifier.train()

        if mode == "partial_finetuning":

            modules = get_finetuning_modules(
                model,
                model_name
            )

            for module in modules:

                module.train()

    return model


def create_model(
    model_name,
    num_classes,
    pretrained=True,
    dropout=0.2,
    mode="feature_extraction",
    device=None
):

    if model_name not in MODEL_REGISTRY:

        raise ValueError(
            f"Modelo no disponible: {model_name}. "
            f"Opciones: {get_available_models()}"
        )

    if num_classes < 2:

        raise ValueError(
            "Se necesitan al menos dos clases."
        )

    if not 0 <= dropout < 1:

        raise ValueError(
            "Dropout debe estar entre 0 y 1."
        )

    if not pretrained and mode != "full_finetuning":

        raise ValueError(
            "Un modelo sin pesos preentrenados "
            "debe entrenarse completamente."
        )

    model_info = MODEL_REGISTRY[model_name]

    constructor = model_info["constructor"]

    weights = (
        model_info["weights"]
        if pretrained
        else None
    )

    # Crear arquitectura
    model = constructor(
        weights=weights
    )

    # Adaptar clasificador
    model = replace_classifier(
        model=model,
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout
    )

    # Configurar parámetros entrenables
    model = configure_trainable_layers(
        model=model,
        model_name=model_name,
        mode=mode
    )

    model.architecture_name = model_name

    model.num_classes = num_classes

    if device is not None:

        model = model.to(device)

    return model


def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = (
        total_params - trainable_params
    )

    trainable_percentage = (
        trainable_params / total_params
    ) * 100

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": trainable_percentage
    }