from collections import OrderedDict
from torch.autograd import grad
import torch
import torch.nn.functional as F


# def smoothen_attributions(attr, kernel_sigma = 0.3, kernel_size = 10):
#     """ Smoothen attributions. Path attribution of shape [..., H, W] where
#         the last 2 dimensions will be spatially smoothened by a Gaussian kernel.
#         kernel_size = size of the smooting kernel in px. Actual smoothing kernel will have size
#                 [kernel_size-1, kernel_size-1]. Padding will be added to keep the original shape.
#         kernel_sigma = std.dev of the gaussian with respect to the kernel_size.
#     """
#     input_sz = [attr.size(-2), attr.size(-1)]
#     X, Y = torch.meshgrid(torch.linspace(0, 1, kernel_size-1), torch.linspace(0, 1, kernel_size-1))
#     coords = torch.stack((X.flatten(), Y.flatten()), dim=0)
#     kernel = torch.exp(-0.5*torch.sum((coords-0.5).pow(2), dim=0)/(kernel_sigma*kernel_sigma))
#     kernel = kernel.reshape(kernel_size-1, kernel_size-1).unsqueeze(0).unsqueeze(0).repeat((attr.size(-3), attr.size(-3), 1, 1))
#     #print(kernel.shape)
#     return torch.conv2d(attr, kernel.to(attr.device), stride=1, padding=(kernel_size//2)-1)

def input_jacobian(net, x):
    """ Compute jacobians w.r.t. input. This function is similar to torch.autograd.functional.jacobian
        in later pytorch versions. 
        Parameters:
            net: Encoder for which encode_deterministic will be called.
        Return the gradient with respect to each input for each sample.
    """
    x.requires_grad_(True)
    out = net.encode_deterministic(images=x)
    J_list = []
    for k in range(net.z_dim):
        J = grad(out[:,k], x, torch.ones_like(out[:,k]), retain_graph=True)[0]
        # print(J.shape)
        J_list.append(J)
    jac = torch.stack(J_list, dim=1)
    return jac

def generator_jacobian(net, z):
    """ Compute jacobians w.r.t. input. This function is similar to torch.autograd.functional.jacobian
        in later pytorch versions. 
    """
    z.requires_grad_(True)
    rec = net.decode(latent=z)
    rec_orgshape = rec.shape
    rec = rec.reshape(len(rec), -1)
    J_list = []
    for k in range(rec.size(1)):
        J = grad(rec[:,k], z, torch.ones_like(rec[:,k]), retain_graph=True)[0]
        #print(J.shape)
        J_list.append(J)
    jac = torch.stack(J_list, dim=1)
    jac = jac.transpose(1,2)
    # print(jac.shape)
    return jac.reshape(jac.size(0), -1 , *rec_orgshape[1:])

# def generator_jacobian_fds(net, z, delta=0.3):
#     """ z: (B, N) latent variables.
#         Approximate derivate by finite differences.
#     """
#     jac_list = []
#     for dim in range(z.size(1)):
#         z_plus = z.clone()
#         z_plus[:,dim] += delta
#         z_minus = z.clone()
#         z_minus[:,dim] -= delta
#         z_in = torch.cat([z_plus, z_minus], dim=0)
#         #print(z_in.shape)
#         rec = net.decode(latent=z_in)
#         rplus, rminus = torch.chunk(rec, 2, dim=0)
#         jac_list.append((rplus-rminus)/(2.0*delta))
#     return torch.stack(jac_list, dim=1)


# def decoder_change(net, x, delta= 0.3):
#     latent = net.encode_deterministic(images=x)
#     # Compute the gradient of the generator.
#     #return generator_jacobian(net, latent)
#     return generator_jacobian_fds(net, latent, delta)
    

def integrated_gradients(net, x, baseline, n_steps = 20):
    """ Return the IG attributions. """
    steps = torch.linspace(0, 1, n_steps).to(net.device)
    baseline = baseline.to(net.device)
    n_samples = len(x)
    if len(baseline.shape) == 3:
        baseline = baseline.unsqueeze(0)

    interps = (x.reshape(-1,x.size(0), x.size(1), x.size(2), x.size(3))*steps.reshape(-1, 1, 1, 1, 1)) + (1-steps.reshape(-1,1,1,1,1))*baseline
    interps = interps.contiguous()
    old_shape = interps.shape
    # print(interps.shape)
    # flatten inputs to compute model gradients.
    interps = interps.reshape(-1, interps.size(2), interps.size(3), interps.size(4))

    jacs = input_jacobian(net, interps)
    #print(jacs.shape)
    ig = jacs.reshape(old_shape[0], old_shape[1], -1, old_shape[2], old_shape[3], old_shape[4]).sum(dim=0)
    return ig

def smoothgrad_gradients(net, x, n_samples = 20, noise_lvl = 0.02):
    """ Return the smoothgrad attributions. """
    noise = noise_lvl*torch.randn(n_samples, *x.shape, device=net.device)
    interps = x.reshape(-1, x.size(0), x.size(1), x.size(2), x.size(3)) + noise
    interps = interps.contiguous()
    old_shape = interps.shape
    #print(interps.shape)
    # flatten inputs to compute model gradients.
    interps = interps.reshape(-1, interps.size(2), interps.size(3), interps.size(4))

    jacs = input_jacobian(net, interps)
    #print(jacs.shape)
    sg = jacs.reshape(old_shape[0], old_shape[1], -1, old_shape[2], old_shape[3], old_shape[4]).sum(dim=0)
    return sg


# # -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
# class _BaseWrapper(object):
#     """
#     Please modify forward() and backward() depending on your task.
#     """
#     def __init__(self, model):
#         super(_BaseWrapper, self).__init__()
#         self.device = model.device
#         self.model = model
#         self.handlers = []  # a set of hook function handlers

#     def generate(self):
#         raise NotImplementedError

#     def forward(self, image):
#         """
#         Simple classification
#         """
#         self.model.zero_grad()
#         # out = self.model.encode_deterministic(images=image)
#         # return out
#         self.logits = self.model(image)
#         self.probs = F.softmax(self.logits, dim=1)
#         return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)


# class GradCam(_BaseWrapper):
#     def __init__(self, model, candidate_layers=[]):
#         super(GradCam, self).__init__(model)
#         self.fmap_pool = OrderedDict()
#         self.grad_pool = OrderedDict()
#         self.candidate_layers = candidate_layers

#         def forward_hook(module, input, output):
#             self.fmap_pool[id(module)] = output.detach()

#         def backward_hook(module, grad_in, grad_out):
#             # print(grad_out[0].shape)
#             self.grad_pool[id(module)] = grad_out[0].detach()

#         for module in self.model.model.named_modules():
#             # print(module[0]) ## print out module names
#             if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
#                 self.handlers.append(module[1].register_forward_hook(forward_hook))
#                 self.handlers.append(module[1].register_backward_hook(backward_hook))

#     def find(self, pool, target_layer):
#         # --- Query the right layer and return it's value.
#         for key, value in pool.items():
#             for module in self.model.model.named_modules():
#                 if id(module[1]) == key:
#                     if module[0] == target_layer:
#                         return value
#         raise ValueError(f"Invalid Layer Name: {target_layer}")

#     def normalize(self, grads):
#         l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
#         return grads /l2_norm

#     def compute_grad_weights(self, grads):
#         grads = self.normalize(grads)
#         return F.adaptive_avg_pool2d(grads, 1)


#     def generate(self, target_layer):
#         fmaps = self.find(self.fmap_pool, target_layer)
#         grads = self.find(self.grad_pool, target_layer)
#         weights = self.compute_grad_weights(grads)

#         gcam = (fmaps * weights).sum(dim=1, keepdim=True)
#         gcam = torch.clamp(gcam, min=0.0)

#         gcam -= gcam.min()
#         gcam /= (gcam.max() + 1e-5)
#         return gcam
    
#     def remove_hooks(self):
#         for fh in self.handlers:
#             fh.remove()


# def grad_cam(net, x):
#     net.model.requires_grad_(True)
#     target_layers = ['encoder.main.10']
#     # target_layers = []
#     gcam = GradCam(model=net, candidate_layers=target_layers)
#     out = net.encode_deterministic(images=x)
#     # fmaps = gcam.find(gcam.fmap_pool, target_layers[0])
#     J_list = []
#     for k in range(net.z_dim):
#         one_hot = torch.zeros((out.shape[0], out.shape[1])).float()
#         one_hot[:,k]=1
#         out.backward(gradient=one_hot.cuda(), retain_graph=True)
#         gcam_out = gcam.generate(target_layers[0])
#         gcam_out = F.upsample(gcam_out, (x.shape[-2], x.shape[-1]), mode='bilinear')
#         J_list.append(gcam_out)
#     jac = torch.stack(J_list, dim=1)
#     gcam.remove_hooks()
#     return jac


# def calc_mean_feature_value(data_loader, iters=50):
#     """ Calculate an average feature to use as a baseline for IG. """
#     x_batch_list = []
#     for i, data in enumerate(data_loader):
#         if i == iters:
#             break
#         _, x, _ = data
#         x_batch_list.append(x)
#     x_batch_list = torch.cat(x_batch_list, dim=0)
#     print(f"Calculating feature mean using {len(x_batch_list)} samples.")
#     return x_batch_list.mean(dim=0)