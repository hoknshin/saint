import torch
import numpy as np


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    elif model.cont_embeddings == 'pos_singleMLP':
        # This is not implemented from authors ... 
        raise Exception('This case should not work!')
        #print (model.simple_MLP)
        #print (x_cont.shape)
        #x_cont_enc = model.simple_MLP[0](x_cont)
    elif model.cont_embeddings == 'temporal_sol':
        x_cont_enc = model.simple_MLP[0](x_cont)
        print (x_cont_enc.shape)
    else:
        raise Exception('This case should not work!')    


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)


    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc




def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_categ,x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
        x1, x2 =  x_categ[index,:], x_cont[index,:]
        x_categ_corr, x_cont_corr = x_categ.clone().detach() ,x_cont.clone().detach()
        x_categ_corr[cat_corr==0] = x1[cat_corr==0]
        x_cont_corr[con_corr==0] = x2[con_corr==0]
        return x_categ_corr, x_cont_corr
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
    else:
        print("yet to write this")

def generate_noise(x_categ,x_cont,noise_params = {'noise_type' : ['gaussian_noise'], 'lambda': 0.1}):
        """Generates noisy version of the samples x
        
        Args:
            x (np.ndarray): Input data to add noise to
        
        Returns:
            (np.ndarray): Corrupted version of input x
            
        """
        device = x_categ.device

        # Dimensions
        batch_size, dim_x_categ = x_categ.shape
        batch_size, dim_x_cont = x_cont.shape
        
        # Get noise type
        noise_type = noise_params["noise_type"]
        noise_level = noise_params["lambda"]

        # Initialize corruption array
#         x_categ_bar = torch.from_numpy(np.zeros(x_categ.shape)).to(device).type(x_categ.dtype)
#         x_cont_bar = torch.from_numpy(np.zeros(x_cont.shape)).to(device).type(x_cont.dtype)
        
        x_categ_bar = np.zeros(x_categ.shape)
        x_cont_bar = np.zeros(x_cont.shape)

        
        # Randomly (and column-wise) shuffle data
        if "swap_noise" in noise_type:
            dtype_x_categ = x_categ.dtype
            dtype_x_cont = x_cont.dtype
            x_categ = x_categ.detach().cpu().numpy()
            x_cont = x_cont.detach().cpu().numpy()
            for i in range(dim_x_cont):
                idx = np.random.permutation(batch_size)
                x_cont_bar[:, i] = x_cont[idx, i]
            for j in range(dim_x_categ):
                idx = np.random.permutation(batch_size)
                x_categ_bar[:, j] = x_categ[idx, j]
            x_cont_corr = x_cont * (1 - noise_level) + x_cont_bar * noise_level
            x_categ_corr = x_categ * (1 - noise_level) + x_categ_bar * noise_level
            x_cont_corr = torch.from_numpy(x_cont_corr).to(device).type(dtype_x_cont)
            x_categ_corr = torch.from_numpy(x_categ_corr).to(device).type(dtype_x_categ)

        # Elif, overwrite x_bar by adding Gaussian noise to x
        elif "gaussian_noise" in noise_type:
            x_cont_bar = torch.from_numpy(np.random.normal(0, noise_level, x_cont.shape)).to(device)
            x_cont_corr = torch.add(x_cont, x_cont_bar.type(x_cont.dtype))
            x_categ_corr = x_categ
        else:
            x_categ_corr = x_categ
            x_cont_corr = x_cont
        
        
        return x_categ_corr,x_cont_corr