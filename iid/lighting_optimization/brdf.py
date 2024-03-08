"""
Adapted from https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/lightnet/models/render/brdf.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def sqrt_(x: torch.Tensor, eps=0) -> torch.Tensor:
    """
    clamping 0 values of sqrt input to avoid NAN gradients
    """
    return torch.sqrt(torch.clamp(x, min=eps))


def reflect(v: torch.Tensor, h: torch.Tensor):
    dot = torch.sum(v * h, dim=2, keepdim=True)
    return 2 * dot * h - v


def square_to_cosine_hemisphere(sample: torch.Tensor):
    u, v = sample[:, :, 0, ...], sample[:, :, 1, ...]
    phi = u * 2 * np.pi
    r = sqrt_(v)
    cos_theta = sqrt_(torch.clamp(1 - v, 0))
    return torch.stack([torch.cos(phi) * r, torch.sin(phi) * r, cos_theta], dim=2)


def get_cos_theta(v: torch.Tensor):
    return v[:, :, 2, ...]


def get_phi(v: torch.Tensor):
    cos_theta = torch.clamp(v[:, :, 2, ...], min=0, max=1)
    sin_theta = torch.clamp(sqrt_(1 - cos_theta * cos_theta), min=1e-8)
    cos_phi = torch.clamp(v[:, :, 0, ...] / sin_theta, -1, 1)
    sin_phi = v[:, :, 1, ...] / sin_theta
    phi = torch.acos(cos_phi)  # (0, pi)
    return torch.where(sin_phi > 0, phi, 2 * np.pi - phi)


def sample_disney_specular(sample: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor):
    """
    :param: sample (bn, spp, 3, h, w)
    :param: roughness (bn, 1, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), phi (bn, spp, h, w), cos theta (bn, spp, h, w)
    """
    # a = torch.clamp(roughness, 0.001)
    a = roughness
    u, v = sample[:, :, 0, ...], sample[:, :, 1, ...]
    phi = u * 2 * np.pi
    cos_theta = sqrt_((1 - v) / (1 + (a * a - 1) * v))
    sin_theta = sqrt_(1 - cos_theta * cos_theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    half = torch.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=2)
    wo = F.normalize(reflect(wi.expand_as(half), half), dim=2, eps=1e-8)
    return wo
    # , phi.squeeze(2), cos_theta.squeeze(2)


def GTR2(ndh, a):
    a2 = a * a
    t = 1.0 + (a2 - 1.0) * ndh * ndh
    return a2 / (np.pi * t * t)


def SchlickFresnel(u):
    m = torch.clamp(1.0 - u, 0, 1)
    return m ** 5


def smithG_GGX(ndv, a):
    a = a * a
    b = ndv * ndv
    return 1.0 / (ndv + sqrt_(a + b - a * b))


def pdf_disney(roughness: torch.Tensor, metallic: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    """
    # specularAlpha = torch.clamp(roughness, 0.001)
    specularAlpha = roughness
    diffuseRatio = 0.5 * (1 - metallic)
    specularRatio = 1 - diffuseRatio
    half = F.normalize(wi + wo, dim=2, eps=1e-8)
    cosTheta = torch.abs(half[:, :, 2, ...])
    pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta
    pdfSpec = pdfGTR2 / torch.clamp(4.0 * torch.abs(torch.sum(wo * half, dim=2)), min=1e-8)
    pdfDiff = torch.abs(wo[:, :, 2, ...]) / np.pi
    pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec
    pdf = torch.where(wi[:, :, 2, ...] < 0, torch.zeros_like(pdf), pdf)
    pdf = torch.where(wo[:, :, 2, ...] < 0, torch.zeros_like(pdf), pdf)
    return pdf


def eval_disney(albedo: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor, wi: torch.Tensor,
                wo: torch.Tensor):
    """
    :param: albedo/roughness/metallic (bn, c, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    """
    h = wi + wo;
    h = F.normalize(h, dim=2, eps=1e-8)

    CSpec0 = torch.lerp(torch.ones_like(albedo) * 0.04, albedo, metallic).unsqueeze(1)

    ldh = torch.clamp(torch.sum((wo * h), dim=2), 0, 1).unsqueeze(2)
    ndv = wi[:, :, 2:3, ...]
    ndl = wo[:, :, 2:3, ...]
    ndh = h[:, :, 2:3, ...]

    FL, FV = SchlickFresnel(ndl), SchlickFresnel(ndv)
    roughness = roughness.unsqueeze(1)
    Fd90 = 0.5 + 2.0 * ldh * ldh * roughness
    Fd = torch.lerp(torch.ones_like(Fd90), Fd90, FL) * torch.lerp(torch.ones_like(Fd90), Fd90, FV)

    Ds = GTR2(ndh, roughness)
    FH = SchlickFresnel(ldh)
    Fs = torch.lerp(CSpec0, torch.ones_like(CSpec0), FH)
    roughg = (roughness * 0.5 + 0.5) ** 2
    Gs1, Gs2 = smithG_GGX(ndl, roughg), smithG_GGX(ndv, roughg)
    Gs = Gs1 * Gs2

    eval_diff = Fd * albedo.unsqueeze(1) * (1.0 - metallic.unsqueeze(1)) / np.pi
    eval_spec = Gs * Fs * Ds
    mask = torch.where(ndl < 0, torch.zeros_like(ndl), torch.ones(ndl))
    return eval_diff, eval_spec, mask


def F_Schlick(SpecularColor, VoH):
    Fc = (1 - VoH) ** 5
    return torch.clamp(50.0 * SpecularColor[:, :, 1:2, ...], min=0, max=1) * Fc + (1 - Fc) * SpecularColor


def GetSpecularEventProbability(SpecularColor, NoV) -> torch.Tensor:
    f = F_Schlick(SpecularColor, NoV);
    return (f[:, :, 0, ...] + f[:, :, 1, ...] + f[:, :, 2, ...]) / 3


def baseColorToSpecularF0(baseColor, metalness):
    return torch.lerp(torch.empty_like(baseColor).fill_(0.04), baseColor, metalness)


def luminance(color):
    if color.size(1) == 1:
        return color
    return color[:, 0:1, ...] * 0.212671 + color[:, 1:2, ...] * 0.715160 + color[:, 2:3, ...] * 0.072169


def probabilityToSampleSpecular(difColor, specColor) -> torch.Tensor:
    lumDiffuse = torch.clamp(luminance(difColor), min=0.01)
    lumSpecular = torch.clamp(luminance(specColor), min=0.01)
    return lumSpecular / (lumDiffuse + lumSpecular)


def shadowedF90(F0):
    t = 1 / 0.04
    return torch.clamp(t * luminance(F0), max=1)


def evalFresnel(f0, f90, NdotS):
    # print(f0.shape, f90.shape, NdotS.shape)
    return f0 + (f90 - f0) * (1 - NdotS) ** 5


def Smith_G1_GGX(alphaSquared, NdotSSquared):
    return 2 / (sqrt_(((alphaSquared * (1 - NdotSSquared)) + NdotSSquared) / NdotSSquared) + 1)


def Smith_G2_GGX(alphaSquared, NdotL, NdotV):
    a = NdotV * sqrt_(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL))
    b = NdotL * sqrt_(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV))
    return 0.5 / (a + b)


def GGX_D(alphaSquared, NdotH):
    b = (alphaSquared - 1) * NdotH * NdotH + 1
    return alphaSquared / (np.pi * b * b)


def pdf_ggx(color: torch.Tensor, roughness: torch.Tensor, metalness: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: color (bn, 3, h, w)
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    :return: pdf (*, *, h, w)
    """
    alpha = roughness * roughness
    alphaSquared = alpha * alpha
    NdotV = wi[:, :, 2, ...]
    h = F.normalize(wi + wo, dim=2, eps=1e-8)
    NdotH = h[:, :, 2, ...]
    pdf_spec = GGX_D(torch.clamp(alphaSquared, min=0.00001), NdotH) * Smith_G1_GGX(alphaSquared, NdotV * NdotV) / (
                4 * NdotV)
    specularF0 = baseColorToSpecularF0(color, metalness)
    diffuseReflectance = color * (1 - metalness)
    kS = probabilityToSampleSpecular(diffuseReflectance, specularF0)
    pdf_diff = wo[:, :, 2, ...] / np.pi
    pdf = kS * pdf_spec + (1 - kS) * pdf_diff
    pdf = torch.where(wi[:, :, 2, ...] <= 0, torch.zeros_like(pdf), pdf)
    pdf = torch.where(wo[:, :, 2, ...] <= 0, torch.zeros_like(pdf), pdf)
    return pdf


def eval_ggx(color: torch.Tensor, roughness: torch.Tensor, metalness: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: color (bn, c, h, w)
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, c, h, w), supposed to be normalized
    :param: wo (*, *, c, h, w), supposed to be normalized
    :return: fr(wi, wo) (*, *, c, h, w)
    """
    NDotL = wo[:, :, 2:3, ...]
    NDotV = wi[:, :, 2:3, ...]
    H = F.normalize(wi + wo, dim=2, eps=1e-8)
    NDotH = H[:, :, 2:3, ...]
    LDotH = torch.sum(wo * H, dim=2, keepdim=True)
    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    D = GGX_D(torch.clamp(alpha2, min=0.00001), NDotH)
    G2 = Smith_G2_GGX(alpha2, NDotL, NDotV)
    specularF0 = baseColorToSpecularF0(color, metalness)
    diffuseReflectance = color * (1 - metalness)
    f = evalFresnel(specularF0.unsqueeze(1), shadowedF90(specularF0).unsqueeze(1), LDotH)
    spec = torch.where(NDotL <= 0, torch.zeros_like(NDotL), f * G2 * D)
    mask = torch.where(NDotL <= 0, torch.zeros_like(NDotL), torch.ones_like(NDotL))
    return diffuseReflectance.unsqueeze(1) / np.pi, spec, mask


def sample_weight_ggx(alphaSquared, NdotL, NdotV):
    G1V = Smith_G1_GGX(alphaSquared, NdotV * NdotV)
    G1L = Smith_G1_GGX(alphaSquared, NdotL * NdotL)
    return G1L / (G1V + G1L - G1V * G1L)


def sample_ggx(sample: torch.Tensor, albedo: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor,
               wi: torch.Tensor):
    """
    :param: sample (bn, spp, 3, h, w)
    :param: roughness (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), weight (bn, spp, 3, h, w)
    """
    specularF0 = baseColorToSpecularF0(albedo, metallic)
    diffuseReflectance = albedo * (1 - metallic)
    kS = probabilityToSampleSpecular(diffuseReflectance, specularF0)
    sample_diffuse = sample[:, :, 2, ...] >= kS

    wo_diff = square_to_cosine_hemisphere(sample[:, :, 1:, ...])
    weight_diff = diffuseReflectance / (1 - kS)
    weight_diff = weight_diff.unsqueeze(1)

    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    # alpha = roughness
    Vh = F.normalize(torch.cat([alpha * wi[:, :, 0:1, ...], alpha * wi[:, :, 1:2, ...], wi[:, :, 2:3, ...]], dim=2),
                     dim=2, eps=1e-8)
    lensq = Vh[:, :, 0:1, ...] ** 2 + Vh[:, :, 1:2, ...] ** 2
    zero_ = torch.zeros_like(Vh[:, :, 0, ...])
    one_ = torch.ones_like(Vh[:, :, 0, ...])
    T1 = torch.where(
        lensq > 0,
        torch.stack([-Vh[:, :, 1, ...], Vh[:, :, 0, ...], zero_], dim=2) / sqrt_(lensq),
        torch.stack([one_, zero_, zero_], dim=2)
    )
    T2 = torch.cross(Vh, T1, dim=2)
    r = sqrt_(sample[:, :, 0:1, ...])
    phi = 2 * np.pi * sample[:, :, 1:2, ...]
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1 + Vh[:, :, 2:3, ...])
    t2 = torch.lerp(sqrt_(1 - t1 ** 2), t2, s)
    Nh = t1 * T1 + t2 * T2 + sqrt_(torch.clamp(1 - t1 * t1 - t2 * t2, min=0)) * Vh
    h = F.normalize(
        torch.cat([alpha * Nh[:, :, 0:1, ...], alpha * Nh[:, :, 1:2, ...], torch.clamp(Nh[:, :, 2:3, ...], min=0)],
                  dim=2), dim=2, eps=1e-8)
    wo = reflect(wi, h)

    HdotL = torch.clamp(torch.sum(h * wo, dim=2, keepdim=True), min=0.00001, max=1.0)
    NdotL = torch.clamp(wo[:, :, 2:3, ...], min=0.00001, max=1.0)
    NdotV = torch.clamp(wi[:, :, 2:3, ...], min=0.00001, max=1.0)
    # NdotH = torch.clamp(h[:,:,2:3,...], min=0.00001, max=1.0)
    # F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL)
    weight = evalFresnel(specularF0, shadowedF90(specularF0), HdotL) * sample_weight_ggx(alpha * alpha, NdotL,
                                                                                         NdotV) / kS.unsqueeze(1)

    wo = torch.where(sample_diffuse.unsqueeze(2), wo_diff, wo)
    weight = torch.where(sample_diffuse.unsqueeze(2), weight_diff, weight)

    return wo, weight


def sample_ggx_specular(sample: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor):
    """
    :param: sample (bn, spp, 2, h, w)
    :param: roughness (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), phi (bn, spp, h, w), cos theta (bn, spp, h, w)
    """
    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    # alpha = roughness
    Vh = F.normalize(torch.cat([alpha * wi[:, :, 0:1, ...], alpha * wi[:, :, 1:2, ...], wi[:, :, 2:3, ...]], dim=2),
                     dim=2, eps=1e-8)
    # bn, spp, _, row, col = Vh.shape
    # Vh = Vh.view(-1, 3, row, col)
    # T1, T2, Vh = utils.hughes_moeller(Vh)
    # T1 = T1.view(bn, spp, 3, row, col)
    # T2 = T2.view(bn, spp, 3, row, col)
    # Vh = Vh.view(bn, spp, 3, row, col)
    lensq = Vh[:, :, 0:1, ...] ** 2 + Vh[:, :, 1:2, ...] ** 2
    zero_ = torch.zeros_like(Vh[:, :, 0, ...])
    one_ = torch.ones_like(Vh[:, :, 0, ...])
    T1 = torch.where(
        lensq > 0,
        torch.stack([-Vh[:, :, 1, ...], Vh[:, :, 0, ...], zero_], dim=2) / sqrt_(lensq),
        torch.stack([one_, zero_, zero_], dim=2)
    )
    T2 = torch.cross(Vh, T1, dim=2)
    r = sqrt_(sample[:, :, 0:1, ...])
    phi = 2 * np.pi * sample[:, :, 1:2, ...]
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1 + Vh[:, :, 2:3, ...])
    t2 = torch.lerp(sqrt_(1 - t1 ** 2), t2, s)
    Nh = t1 * T1 + t2 * T2 + sqrt_(torch.clamp(1 - t1 * t1 - t2 * t2, min=0)) * Vh
    h = F.normalize(
        torch.cat([alpha * Nh[:, :, 0:1, ...], alpha * Nh[:, :, 1:2, ...], torch.clamp(Nh[:, :, 2:3, ...], min=0)],
                  dim=2), dim=2, eps=1e-8)
    wo = reflect(wi, h)
    return wo

def pdf_diffuse(wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: color (bn, 3, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    :return: pdf (*, *, h, w)
    """
    pdf = wo[:,:,2,...] / np.pi
    pdf = torch.where(wi[:,:,2,...] <= 0, torch.zeros_like(pdf), pdf)
    pdf = torch.where(wo[:,:,2,...] <= 0, torch.zeros_like(pdf), pdf)
    return pdf


def eval_diffuse(color: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: color (bn, c, h, w)
    :param: wi (*, *, c, h, w), supposed to be normalized
    :param: wo (*, *, c, h, w), supposed to be normalized
    :return: fr(wi, wo) (*, *, c, h, w)
    """
    NDotL = wo[:,:,2:3,...]
    NDotV = wi[:,:,2:3,...]
    H = F.normalize(wi + wo, dim=2, eps=1e-8)
    NDotH = H[:,:,2:3,...]
    LDotH = torch.sum(wo*H, dim=2, keepdim=True)

    diffuseReflectance = color.unsqueeze(1) / np.pi
    spec = torch.zeros_like(diffuseReflectance)
    mask = torch.where(NDotL <= 0, torch.zeros_like(NDotL), torch.ones_like(NDotL))
    return diffuseReflectance, spec, mask
