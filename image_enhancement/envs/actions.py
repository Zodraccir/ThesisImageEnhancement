import torch
import torch.utils
import torch.utils.data
#from ColorAlgorithms import saturation, hue
import torchvision



def select(img, act):
    img = img.detach().clone().unsqueeze_(0)
    if act == 0:
        return gamma_corr(img, 0.6, 0).squeeze()
    elif act == 1:
        return gamma_corr(img, 0.6, 1).squeeze()
    elif act == 2:
        return gamma_corr(img, 0.6, 2).squeeze()
    elif act == 3:
        return gamma_corr(img, 1.1, 0).squeeze()
    elif act == 4:
        return gamma_corr(img, 1.1, 1).squeeze()
    elif act == 5:
        return gamma_corr(img, 1.1, 2).squeeze()
    elif act == 6:
        return gamma_corr(img, 0.6).squeeze()
    elif act == 7:
        return gamma_corr(img, 1.1).squeeze()
    elif act == 8:
        return brightness(img, 0.1, 0).squeeze()
    elif act == 9:
        return brightness(img, 0.1, 1).squeeze()
    elif act == 10:
        return brightness(img, 0.1, 2).squeeze()
    elif act == 11:
        return brightness(img, -0.1, 0).squeeze()
    elif act == 12:
        return brightness(img, -0.1, 1).squeeze()
    elif act == 13:
        return brightness(img, -0.1, 2).squeeze()
    elif act == 14:
        return brightness(img, 0.1).squeeze()
    elif act == 15:
        return brightness(img, -0.1).squeeze()
    elif act == 16:
        return contrast(img, 0.8, 0).squeeze()
    elif act == 17:
        return contrast(img, 0.8, 1).squeeze()
    elif act == 18:
        return contrast(img, 0.8, 2).squeeze()
    elif act == 19:
        return contrast(img, 2, 0).squeeze()
    elif act == 20:
        return contrast(img, 2, 1).squeeze()
    elif act == 21:
        return contrast(img, 2, 2).squeeze()
    elif act == 22:
        return contrast(img, 0.8).squeeze()
    elif act == 23:
        return contrast(img, 2).squeeze()
    elif act == 24:
        return torchvision.transforms.functional.adjust_saturation(img, 0.5).squeeze()
    elif act == 25:
        return torchvision.transforms.functional.adjust_saturation(img, 2).squeeze()
    elif act == 26:
        return torchvision.transforms.functional.adjust_hue(img, 0.05).squeeze()
    elif act == 27:
        return torchvision.transforms.functional.adjust_hue(img, -0.05).squeeze()
    elif act == 28:
        return img.squeeze()


def select_fine(img, act):
    img = img.detach().clone().unsqueeze_(0)
    if act == 0:
        return gamma_corr(img, 0.775, 0).squeeze()
    elif act == 1:
        return gamma_corr(img, 0.775, 1).squeeze()
    elif act == 2:
        return gamma_corr(img, 0.775, 2).squeeze()
    elif act == 3:
        return gamma_corr(img, 1.05, 0).squeeze()
    elif act == 4:
        return gamma_corr(img, 1.05, 1).squeeze()
    elif act == 5:
        return gamma_corr(img, 1.05, 2).squeeze()
    elif act == 6:
        return gamma_corr(img, 0.775).squeeze()
    elif act == 7:
        return gamma_corr(img, 1.05).squeeze()
    elif act == 8:
        return brightness(img, 0.05, 0).squeeze()
    elif act == 9:
        return brightness(img, 0.05, 1).squeeze()
    elif act == 10:
        return brightness(img, 0.05, 2).squeeze()
    elif act == 11:
        return brightness(img, -0.05, 0).squeeze()
    elif act == 12:
        return brightness(img, -0.05, 1).squeeze()
    elif act == 13:
        return brightness(img, -0.05, 2).squeeze()
    elif act == 14:
        return brightness(img, 0.05).squeeze()
    elif act == 15:
        return brightness(img, -0.05).squeeze()
    elif act == 16:
        return contrast(img, 0.894, 0).squeeze()
    elif act == 17:
        return contrast(img, 0.894, 1).squeeze()
    elif act == 18:
        return contrast(img, 0.894, 2).squeeze()
    elif act == 19:
        return contrast(img, 1.414, 0).squeeze()
    elif act == 20:
        return contrast(img, 1.414, 1).squeeze()
    elif act == 21:
        return contrast(img, 1.414, 2).squeeze()
    elif act == 22:
        return contrast(img, 0.894).squeeze()
    elif act == 23:
        return contrast(img, 1.414).squeeze()
    elif act == 24:
        return torchvision.transforms.functional.adjust_saturation(img, 0.707).squeeze()
    elif act == 25:
        return torchvision.transforms.functional.adjust_saturation(img, 1.414).squeeze()
    elif act == 26:
        return torchvision.transforms.functional.adjust_hue(img, 0.025).squeeze()
    elif act == 27:
        return torchvision.transforms.functional.adjust_hue(img, -0.025).squeeze()
    elif act == 28:
        return img.squeeze()


def gamma_corr(image, gamma, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = mod[:, channel, :, :] ** gamma
    else:
        mod = mod ** gamma
    return mod


def brightness(image, bright, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
    else:
        mod = torch.clamp(mod + bright, 0, 1)
    return mod


def contrast(image, alpha, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(
            torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])), 0,
            1)
    else:
        mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
    return mod