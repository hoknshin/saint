import torch
from torch import nn
import torchsummary
from models import SAINT

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np
from tqdm.auto import trange, tqdm

from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

initialize(version_base=None, config_path="../config", job_name="test_app")
opt = compose(config_name="mnist")
print(OmegaConf.to_yaml(opt))
OmegaConf.set_struct(opt, True)

modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
with open_dict(opt):
    if opt.task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    if opt.pretrain:
        wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task=='multiclass':
            wandb.init(project="saint_v2_all_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
   


print('Downloading and processing the dataset, it might take some time.')
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(f'number of features: {nfeat}, batch size: {opt.batchsize}')
print ('[Hyperparameters]')
print(opt)

if opt.active_log:
    wandb.config.update(opt)
print ('Creating data loaders...')
train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

print ('Creating SAINT model ...')
model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim,
    dim_head=opt.dim_head,
)
vision_dset = opt.vision_dset
model.to(device)
#print (model)

for data in trainloader:
    break
x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
_ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)           
print (f'x_categ, x_cont, y_gts, cat_mask, con_mask')
print (x_categ.shape, x_cont.shape, y_gts.shape, cat_mask.shape, con_mask.shape)
print ('x_categ_enc, x_cont_enc')
print (x_categ_enc.shape, x_cont_enc.shape)

# input_size = [x_categ_enc, x_cont_enc]
# total_tokens = len(cat_dims) + 0 # num_special_tokens
cat_input_size = (len(cat_dims), opt.embedding_size)
if opt.cont_embeddings == 'MLP':
    cont_input_size = (len(con_idxs), opt.embedding_size)
torchsummary.summary(model, input_size=[cat_input_size, cont_input_size])


if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'




if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')
for epoch in trange(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs,y_gts) 
        else:
            loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    if opt.active_log:
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
        'loss': loss.item()
        })
    if epoch%5==0:
            model.eval()
            with torch.no_grad():
                if opt.task in ['binary','multiclass']:
                    accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                    test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        (epoch + 1, test_accuracy,test_auroc ))
                    if opt.active_log:
                        wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                    if opt.task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    else:
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                        # if auroc > best_valid_auroc:
                        #     best_valid_auroc = auroc
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy               
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                else:
                    valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                    test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                    print('[EPOCH %d] VALID RMSE: %.3f' %
                        (epoch + 1, valid_rmse ))
                    print('[EPOCH %d] TEST RMSE: %.3f' %
                        (epoch + 1, test_rmse ))
                    if opt.active_log:
                        wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()
                


total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' %(best_test_rmse))

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse , 
        'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
    else:
        wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
        'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
