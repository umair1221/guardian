# %% [markdown]
# # Importing Libraries

# %%
import os
import sys
import time
import torch 
import random
import network
import argparse
import platform
import ivtmetrics # You must "pip install ivtmetrics" to use
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np
# import lpips
import piq
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import optuna
from optuna.trial import TrialState
import wandb

from torch.cuda.amp import GradScaler, autocast

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# export CUDA_VISIBLE_DEVICES=1

# %% [markdown]
# # Argument Parser
# ## In case of running using cmd

# %%
#%% @args parsing
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'], help='Model name?')
parser.add_argument('--version', type=int, default=0,  help='Model version control (for keeping several versions)') 
parser.add_argument('--hr_output', action='store_true', help='Whether to use higher resolution output (32x56) or not (8x14). Default: False')
parser.add_argument('--use_ln', action='store_true', help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ') 
# job
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test',  action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,  help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/path/to/dataset', help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval', choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval', 'cholect45-crossval'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,  choices=[1,2,3,4,5,], help='The test split in k-fold cross-validation')
parser.add_argument('--image_width', type=int, default=448, help='Image width ')  
parser.add_argument('--image_height', type=int, default=256, help='Image height ')  
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')  
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*', default=['original', 'vflip', 'hflip', 'contrast', 'rot90'], help='List augumentation styles (see dataloader.py for list of supported styles).')
# hp
parser.add_argument('-b', '--batch', type=int, default=32,  help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100,  help='How many training epochs?')
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[9,18,58], help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01], help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5,  help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10,  help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99,  help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95,  help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1,  help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
# device
parser.add_argument('--gpu', type=str, default="0",  help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()
import torch

# %% [markdown]
# # Parameters Definition

# %%
#%% @params definitions
is_train        = FLAGS.train
is_test         = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir        = FLAGS.data_dir
kfold           = FLAGS.kfold if "crossval" in dataset_variant else 0
version         = FLAGS.version
hr_output       = FLAGS.hr_output
use_ln          = FLAGS.use_ln
batch_size      = FLAGS.batch
pretrain_dir    = FLAGS.pretrain_dir
test_ckpt       = FLAGS.test_ckpt
weight_decay    = FLAGS.weight_decay
learning_rates  = FLAGS.initial_learning_rates
warmups         = FLAGS.warmups
decay_steps     = FLAGS.decay_steps
decay_rate      = FLAGS.decay_rate
power           = FLAGS.power
momentum        = FLAGS.momentum
epochs          = FLAGS.epochs
gpu             = FLAGS.gpu
image_height    = FLAGS.image_height
image_width     = FLAGS.image_width
image_channel   = FLAGS.image_channel
num_triplet     = FLAGS.num_triplet_classes
num_tool        = FLAGS.num_tool_classes
num_verb        = FLAGS.num_verb_classes
num_target      = FLAGS.num_target_classes
val_interval    = FLAGS.epochs-1 if FLAGS.val_interval==-1 else FLAGS.val_interval
set_chlg_eval   = True if "challenge" in dataset_variant else False # To observe challenge evaluation protocol
gpu             = ",".join(str(FLAGS.gpu).split(","))
decodelayer     = FLAGS.decoder_layer
addnorm         = "layer" if use_ln else "batch"
modelsize       = "high" if hr_output else "low"
FLAGS.multigpu  = len(gpu) > 1  # not yet implemented !
mheaders        = ["","l", "cholect", "k"]
margs           = [FLAGS.model, decodelayer, dataset_variant, kfold]
wheaders        = ["norm", "res"]
wargs           = [addnorm, modelsize]
modelname       = "_".join(["{}{}".format(x,y) for x,y in zip(mheaders, margs) if len(str(y))])+"_"+\
                  "_".join(["{}{}".format(x,y) for x,y in zip(wargs, wheaders) if len(str(x))])
model_dir       = "./__checkpoint__/run_{}".format(version)

if not os.path.exists(model_dir): os.makedirs(model_dir)
resume_ckpt     = None
ckpt_path       = os.path.join(model_dir, '{}.pth'.format(modelname))
logfile         = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations      = FLAGS.augmentation_list 
iterable_augmentations  = []
print("Configuring network ...")

#%% @functions (helpers)
def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' 
    

def get_weight_balancing(case='cholect50'):
    # 50:   cholecT50, data splits as used in rendezvous paper
    # 50ch: cholecT50, data splits as used in CholecTriplet challenge
    # 45cv: cholecT45, official data splits (cross-val)
    # 50cv: cholecT50, official data splits (cross-val)
    switcher = {
        'cholect50': {
            'tool'  :   [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
            'verb'  :   [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636, 12.13692946, 0.40462028],
            'target':   [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446, 0.285756500, 12.72368421, 0.6250808,  3.85771277, 6.95683453, 0.84923888, 0.40130032]
        },
        'cholect50-challenge': {
            'tool':     [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
            'verb':     [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 1.28428410, 7.35822511, 18.67857143, 0.45704490],
            'target':   [0.07333818, 0.87139287, 0.42853950, 1.00000000, 17.67281106, 13.94545455, 1.44880997, 6.04889590, 0.326188650, 16.82017544, 0.63577586, 6.79964539, 6.19547658, 0.96284208, 0.51559559]
        },
        'cholect45-crossval': {
            1: {
                'tool':     [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                'verb':     [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889, 17.07784431, 0.45241117],
                'target':   [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319, 0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
                },
            2: {
                'tool':     [0.0854156, 0.89535362, 0.10995253, 2.74936869, 1.78264429, 1.13234529],
                'verb':     [0.36346863, 0.06771776, 0.07893261, 0.82842725, 1.33892161, 2.13049748, 1.26120359, 5.72674419, 19.7, 0.43189126],
                'target':   [0.07530655, 0.97961957, 0.4325135, 0.99393438, 15.5387931, 14.5951417, 1.53862569, 6.01836394, 0.35184462, 15.81140351, 0.709506, 5.79581994, 8.08295964, 1.0, 0.52689272]
            },
            3: {
                "tool" :   [0.0915228, 0.89714969, 0.12057004, 2.72128174, 1.94092281, 1.12948557],
                "verb" :   [0.43636862, 0.07558554, 0.0891017, 0.81820519, 1.53645582, 2.31924198, 1.28565657, 6.49387755, 18.28735632, 0.48676763],
                "target" : [0.06841828, 0.90980736, 0.38826607, 1.0, 14.3640553, 12.9875, 1.25939394, 5.38341969, 0.29060227, 13.67105263, 0.59168565, 6.58985201, 5.72977941, 0.86824513, 0.47682423]

            },
            4: {
                'tool':     [0.08222218, 0.85414117, 0.10948695, 2.50868784, 1.63235867, 1.20593318],
                'verb':     [0.41154261, 0.0692142, 0.08427214, 0.79895288, 1.33625219, 2.2624166, 1.35343681, 7.63, 17.84795322, 0.43970609],
                'target':   [0.07536126, 0.85398445, 0.4085784, 0.95464422, 15.90497738, 18.5978836, 1.55875831, 5.52672956, 0.33700863, 15.41666667, 0.74755423, 5.4921875, 6.11304348, 1.0, 0.50641118],
            },
            5: {
                'tool':     [0.0804654, 0.92271157, 0.10489631, 2.52302243, 1.60074906, 1.09141982],
                'verb':     [0.50710436, 0.06590258, 0.07981184, 0.81538866, 1.29267277, 2.20525568, 1.29699248, 7.32311321, 25.45081967, 0.46733895],
                'target':   [0.07119395, 0.87450495, 0.43043372, 0.86465981, 14.01984127, 23.7114094, 1.47577277, 5.81085526, 0.32129865, 22.79354839, 0.63304067, 6.92745098, 5.88833333, 1.0, 0.53175798]
            }
        },
        'cholect50-crossval': {
            1:{
                'tool':     [0.0828851, 0.8876, 0.10830995, 2.93907285, 1.63884786, 1.14499484],
                'verb':     [0.29628942, 0.07366916, 0.08267971, 0.83155428, 1.25402434, 2.38358209, 1.34938741, 7.56872038, 12.98373984, 0.41502079],
                'target':   [0.06551745, 1.0, 0.36345711, 0.82434783, 13.06299213, 8.61818182, 1.4017744, 4.62116992, 0.32822238, 45.45205479, 0.67343211, 4.13200498, 8.23325062, 0.88527215, 0.43113306],

            },
            2:{
                'tool':     [0.08586283, 0.87716737, 0.11068887, 2.84210526, 1.81016949, 1.16283571],
                'verb':     [0.30072757, 0.07275414, 0.08350168, 0.80694143, 1.39209979, 2.22754491, 1.31448763, 6.38931298, 13.89211618, 0.39397505],
                'target':   [0.07056703, 1.0, 0.39451115, 0.91977006, 15.86206897, 9.68421053, 1.44483706, 5.44378698, 0.31858714, 16.14035088, 0.7238395, 4.20571429, 7.98264642, 0.91360477, 0.43304307],
            },
            3:{
            'tool':      [0.09225068, 0.87856006, 0.12195811, 2.82669323, 1.97710987, 1.1603972],
                'verb':     [0.34285159, 0.08049804, 0.0928239, 0.80685714, 1.56125608, 2.23984772, 1.31471136, 7.08835341, 12.17241379, 0.43180428],
                'target':   [0.06919395, 1.0, 0.37532866, 0.9830703, 15.78801843, 8.99212598, 1.27597765, 5.36990596, 0.29177312, 15.02631579, 0.64935557, 5.08308605, 5.86643836, 0.86580743, 0.41908257], 
            },
            4:{
                'tool':     [0.08247885, 0.83095539, 0.11050268, 2.58193042, 1.64497676, 1.25538881],
                'verb':     [0.31890981, 0.07380354, 0.08804592, 0.79094077, 1.35928144, 2.17017208, 1.42947103, 8.34558824, 13.19767442, 0.40666428],
                'target':   [0.07777646, 0.95894072, 0.41993829, 0.95592153, 17.85972851, 12.49050633, 1.65701092, 5.74526929, 0.33763901, 17.31140351, 0.83747083, 3.95490982, 6.57833333, 1.0, 0.47139615],
            },
            5:{
                'tool':     [0.07891691, 0.89878025, 0.10267677, 2.53805556, 1.60636428, 1.12691169],
                'verb':     [0.36420961, 0.06825313, 0.08060635, 0.80956984, 1.30757221, 2.09375, 1.33625848, 7.9009434, 14.1350211, 0.41429631],
                'target':   [0.07300329, 0.97128713, 0.42084942, 0.8829883, 15.57142857, 19.42574257, 1.56521739, 5.86547085, 0.32732733, 25.31612903, 0.70171674, 4.55220418, 6.13125, 1.0, 0.48528321],
            }
        }
    }
    return switcher.get(case)
     

# %% [markdown]
# # Loading and Building Model

# %%
# Path to model checkpoint
test_ckpt = 'weights/rendezvous_l8_cholectcholect45-crossval_k1_batchnorm_lowres_180.pth'

# Load base structure model
model = network.Rendezvous('resnet18', hr_output=hr_output, use_ln=use_ln).cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


#%% performance tracker for hp tuning
benchmark   = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
print("Model built ...")

# ckpt_path = os.path.join(model_dir, '{}.pth'.format(modelname))

# Load the checkpoint of the trained model
if os.path.exists(test_ckpt):
    model.load_state_dict(torch.load(test_ckpt) ,  strict=False)
    # model= nn.DataParallel(model)
    print("Trained Model Loaded Successfully...!")

# %% [markdown]
# # Define Loss, Activation, and mAP Metrics

# %%
# Or constant weights from average of the random sampling of the dataset: we found this to produce better result.
tool_weight     = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
verb_weight     = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448, 2.69230769, 0.60062402]
target_weight   = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034, 0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]

#%% performance tracker for hp tuning
benchmark   = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

#%% Loss
activation  = nn.Sigmoid()
loss_fn_i   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).cuda())
loss_fn_v   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).cuda())
loss_fn_t   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).cuda())
loss_fn_ivt = nn.BCEWithLogitsLoss()


#%% evaluation metrics
mAP = ivtmetrics.Recognition(100)
mAP.reset_global()
if not set_chlg_eval:
    mAPi = ivtmetrics.Recognition(6)
    mAPv = ivtmetrics.Recognition(10)
    mAPt = ivtmetrics.Recognition(15)
    mAPi.reset_global()
    mAPv.reset_global()
    mAPt.reset_global()

#%% Adversarial metrics
mAP_adv = ivtmetrics.Recognition(100)
mAP_adv.reset_global()
if not set_chlg_eval:
    mAPi_adv = ivtmetrics.Recognition(6)
    mAPv_adv = ivtmetrics.Recognition(10)
    mAPt_adv = ivtmetrics.Recognition(15)
    mAPi_adv.reset_global()
    mAPv_adv.reset_global()
    mAPt_adv.reset_global()
print("Metrics built ...")

# %% [markdown]
# # Dataset Loading

# %%
#%% data loading : variant and split selection (Note: original paper used different augumentation per epoch)
# data_dir = '/share/sdb/umairnawaz/Data/'
# dataset_variant= 'cholect45-crossval'
dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, 
            dataset_variant=dataset_variant,
            test_fold=kfold,
            augmentation_list=data_augmentations,
            )

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()

# %% [markdown]
# ## Load only test dataset in dataloader

# %%
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3*batch_size, num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
 

test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=3*batch_size, num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
print("Dataset loaded ...")

# %%
# Function to perform PGD attack

def projected_gradient_descent(model, x, y, num_steps=10, alpha=2/255, eps=8/255, random_start=True):
    # Adversarial Attack Images
    x_adv = x.clone().detach().requires_grad_(True)

    # Original Images
    x = x.clone().detach().cuda()

    # Initiate attack with random adversarial image
    if random_start:
        # Start with uniformly random point
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps , eps)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    
    targeted = False  # Change to True if we have a targeted attack
    y_target = None    # Set the target label if it's a targeted attack
    for i in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)  # Create a new leaf variable

        # Add additional layer to normalize the input to expected input shape of model
        # resnet_model = nn.Sequential(
        #     normalize,
        #     model
        # ).to(device)

        # min_value = x_adv.min().item()
        # max_value = x_adv.max().item()

        # print("Min and Max Values are: " , min_value , max_value)
        # print(k)
        
        model.eval()
        for param in model.parameters(): param.grad = None
        # Forward pass
        tool, verb, target, triplet = model(x_adv)

        # Extracting Logits and CAMs of each category
        
        cam_i, logit_i  = tool
        cam_v, logit_v  = verb
        cam_t, logit_t  = target
        logit_ivt       = triplet            

        # Compute Loss
        loss_i          = loss_fn_i(logit_i, y[0].float())
        loss_v          = loss_fn_v(logit_v, y[1].float())
        loss_t          = loss_fn_t(logit_t, y[2].float())
        loss_ivt        = loss_fn_ivt(logit_ivt, y[3].float())  

        # print(loss_i)
        # Total Loss
        loss            = (loss_i) + (loss_v) + (loss_t) + loss_ivt 

        # Find the gradient
        grad = torch.autograd.grad(
            loss, x_adv, retain_graph=False, create_graph=False, allow_unused=True
        )[0]

        if grad is not None:
            # Apply attack using gradient and hyper-params
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, min=0, max=1).detach()

    # print("Final Shape: " , x_adv.shape)
    return x_adv

# %%
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)



# %%


# %%
# Normalization layer to normalize the input to expected input of model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Device to load the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
                normalize,
                model
            ).to(device)
model.train()
# Set BIM parameters
num_steps = 1
alpha = 4/255
eps = 8/255





# Apply attack on the test dataloader
def train(dataloader, activation, loss_fn_i, loss_fn_v, loss_fn_t, loss_fn_ivt, optimizers, scheduler, final_eval=False):
    start = time.time()
    print("DataLoader Started:")
    if final_eval:
        print("final_eval")
    run_loss = AverageMeter()
    
    train_loss = 0
    correct = 0
    total = 0
    for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
        # print(f"Counter: {i}")
        # i += 1
        img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        for param in model.parameters(): param.grad = None
        # Get Adv Samples
        
        # resnet_model = nn.Sequential(
        #             normalize,
        #             model
        #         ).to(device)
        # resnet_model = resnet_model.eval()
        adv_example = projected_gradient_descent(model, img, (y1, y2, y3, y4),
                                                 num_steps, alpha, eps,random_start = True )
        # print("Adversarial Done..")
        model.train()
        # for param in model.parameters(): param.grad = None
        
        # Perform inference on adv samples
        tool, verb, target, triplet = model(adv_example)
        # Shall we keep it?
        # with autocast(enabled=args.amp):
        
        cam_i, logit_i  = tool
        cam_v, logit_v  = verb
        cam_t, logit_t  = target
        logit_ivt       = triplet                
        loss_i          = loss_fn_i(logit_i, y1.float())
        loss_v          = loss_fn_v(logit_v, y2.float())
        loss_t          = loss_fn_t(logit_t, y3.float())
        loss_ivt        = loss_fn_ivt(logit_ivt, y4.float())  
        loss            = (loss_i) + (loss_v) + (loss_t) + loss_ivt 

        for param in model.parameters(): param.grad = None

        # Should we uncomment this? Because in main training it is used
        # for param in model.parameters():
        #     param.grad = None

        
        loss.backward()
        for opt in optimizers:
            opt.step()
        # train_loss += loss.item()
        # run_loss.update(loss.item(), n=batch_size)
        # if batch % 10 == 0:
        #     print('\nCurrent batch:', str(batch))
        #     print('Current adversarial train loss:', run_loss.avg)
    # wandb.log({ "Total_Loss": loss })
    # learning rate schedule update
    for sch in scheduler:
        sch.step()
    # for param in model.parameters(): param.grad = None
    # for param in resnet_model.parameters(): param.grad = None
    print(f'completed | Losses =>total: [{loss.item():.4f}] i: [{loss_i.item():.4f}] v: [{loss_v.item():.4f}] t: [{loss_t.item():.4f}] ivt: [{loss_ivt.item():.4f}] >> eta: {(time.time() - start):.2f} secs', file=open(logfile, 'a+'))
    # return run_loss.avg


# %%
def test(dataloader, activation, final_eval=False):
    start = time.time()
    mAP.reset()  
    if final_eval and not set_chlg_eval:
        mAPv.reset() 
        mAPt.reset() 
        mAPi.reset()
    
    # Adversarial Metrics
    mAP_adv.reset()  
    if final_eval and not set_chlg_eval:
        mAPv_adv.reset() 
        mAPt_adv.reset() 
        mAPi_adv.reset()

    model.eval()
    print("DataLoader Started:")
    # with torch.no_grad():
    for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
        img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        
        # resnet_model = nn.Sequential(
        #         normalize,
        #         model
        #     ).to(device)
        # resnet_model = resnet_model.eval()
        #################### Normal Testing #####################
        # img = denormalize_from_01(img, img_min, img_max):
        with torch.no_grad():    
            tool, verb, target, triplet = model(img)
        
        if final_eval:
            cam_i, logit_i = tool
            cam_v, logit_v = verb
            cam_t, logit_t = target
            mAPi.update(y1.float().detach().cpu(), activation(logit_i).detach().cpu()) # Log metrics 
            mAPv.update(y2.float().detach().cpu(), activation(logit_v).detach().cpu()) # Log metrics 
            mAPt.update(y3.float().detach().cpu(), activation(logit_t).detach().cpu()) # Log metrics 

        mAP.update(y4.float().detach().cpu(), activation(triplet).detach().cpu()) # Log metrics 


        ############# Adversarial Attack Started ##############
        # img_min , img_max = img.min() , img.max()
        # img = normalize_to_01(img)
        adversarial_example = projected_gradient_descent(model, img, (y1, y2, y3, y4),
                                                num_steps, alpha, eps,random_start = True )
        # for i , img_adv in enumerate(adversarial_example):
        #     save_image(img[i], f'Images-Attack/Clean/img{i}-{batch}.png')
        #     save_image(img_adv, f'Images-Attack/Adv/img-adv{i}-{batch}.png')

        #### Finding Metrics for Image Quality ####

        # adversarial_example = normalize_to_01(adversarial_example)
        # img = normalize_to_01(img)
        # psnr_values_list.append(piq.psnr(adversarial_example, img).item())
        # ssim_values_list.append(piq.ssim(adversarial_example, img).item())
        # lpips_values_list.append(lpips(2*adversarial_example-1, 2*img-1).item())
        
        # break
        

        ######################## Adversarial Testing ########################
        with torch.no_grad():
            tool_adv, verb_adv, target_adv, triplet_adv = model(adversarial_example)
        
        # _, predicted_adv_tool = torch.max(tool_adv[1].data, 1)  # Assuming tool_adv[1] contains the logits
        # _, predicted_adv_verb = torch.max(verb_adv[1].data, 1)  # Assuming verb_adv[1] contains the logits
        # _, predicted_adv_target = torch.max(target_adv[1].data, 1)  # Assuming target_adv[1] contains the logits
        
        if final_eval:
            cam_i, logit_i_adv = tool_adv
            cam_v, logit_v_adv = verb_adv
            cam_t, logit_t_adv = target_adv
            mAPi_adv.update(y1.float().detach().cpu(), activation(logit_i_adv).detach().cpu()) # Log metrics 
            mAPv_adv.update(y2.float().detach().cpu(), activation(logit_v_adv).detach().cpu()) # Log metrics 
            mAPt_adv.update(y3.float().detach().cpu(), activation(logit_t_adv).detach().cpu()) # Log metrics 
        mAP_adv.update(y4.float().detach().cpu(), activation(triplet_adv).detach().cpu()) # Log metrics 

    
    wandb.log({"mAP": mAP.compute_video_AP()['mAP'] , "mAP_adv": mAP_adv.compute_video_AP()['mAP'] })

    print("Test Results for Clean Samples: | eta {:.2f} secs | mAP => i: [{:.5f}] | mAP => v: [{:.5f}] | mAP => t: [{:.5f}] | mAP => ivt: [{:.5f}] ".format( (time.time() - start), mAP.compute_video_AP('i', ignore_null=set_chlg_eval)['mAP'] , mAP.compute_video_AP('v', ignore_null=set_chlg_eval)['mAP'] , mAP.compute_video_AP('t', ignore_null=set_chlg_eval)['mAP'] ,  mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP']), file=open(logfile, 'a+'))
    print("Test Results for Adversarial Samples: | eta {:.2f} secs | mAP => i: [{:.5f}] | mAP => v: [{:.5f}] | mAP => t: [{:.5f}] | mAP => ivt: [{:.5f}] ".format( (time.time() - start), mAP_adv.compute_video_AP('i', ignore_null=set_chlg_eval)['mAP'] , mAP_adv.compute_video_AP('v', ignore_null=set_chlg_eval)['mAP'] , mAP_adv.compute_video_AP('t', ignore_null=set_chlg_eval)['mAP'] ,  mAP_adv.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP']), file=open(logfile, 'a+'))
    
    mAP.video_end() 
    mAP_adv.video_end() 

    if final_eval:
        mAPv.video_end()
        mAPt.video_end()
        mAPi.video_end()
        
    if final_eval:
        mAPv_adv.video_end()
        mAPt_adv.video_end()
        mAPi_adv.video_end()

# %%
# for epoch in range(0, 200):
#     # adjust_learning_rate(optimizer, epoch)
#     train(epoch)
#     test(epoch)

# %%
#%% optimizer and lr scheduler
# wp_lr           = [lr/power for lr in learning_rates]

# module_i        = list(set(model[1].parameters()) - set(model[1].encoder.cagam.parameters()) - set(model[1].encoder.bottleneck.parameters()) - set(model[1].decoder.parameters()))
# module_ivt      = list(set(model[1].encoder.bottleneck.parameters()).union(set(model[1].decoder.parameters())))
# module_vt       = model[1].encoder.cagam.parameters()

# optimizer_i     = torch.optim.SGD(module_i, lr=wp_lr[0], weight_decay=weight_decay)
# scheduler_ia    = torch.optim.lr_scheduler.LinearLR(optimizer_i, start_factor=power, total_iters=warmups[0])
# scheduler_ib    = torch.optim.lr_scheduler.ExponentialLR(optimizer_i, gamma=decay_rate)
# scheduler_i     = torch.optim.lr_scheduler.SequentialLR(optimizer_i, schedulers=[scheduler_ia, scheduler_ib], milestones=[warmups[0]+1])

# optimizer_vt    = torch.optim.SGD(module_vt, lr=wp_lr[1], weight_decay=weight_decay)
# scheduler_vta   = torch.optim.lr_scheduler.LinearLR(optimizer_vt, start_factor=power, total_iters=warmups[1])
# scheduler_vtb   = torch.optim.lr_scheduler.ExponentialLR(optimizer_vt, gamma=decay_rate)
# scheduler_vt    = torch.optim.lr_scheduler.SequentialLR(optimizer_vt, schedulers=[scheduler_vta, scheduler_vtb], milestones=[warmups[1]+1])

# optimizer_ivt   = torch.optim.SGD(module_ivt, lr=wp_lr[2], weight_decay=weight_decay)
# scheduler_ivta  = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
# scheduler_ivtb  = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
# scheduler_ivt   = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb], milestones=[warmups[2]+1])

# lr_schedulers   = [scheduler_i, scheduler_vt, scheduler_ivt]
# optimizers      = [optimizer_i, optimizer_vt, optimizer_ivt]

# %%
# epochs = 2

def weight_mgt(score, epoch):
    # hyperparameter selection based on validation set
    global benchmark
    if score > benchmark.item():
        torch.save(model.state_dict(), ckpt_path)
        benchmark = score
        print(f'>>> Saving checkpoint for epoch {epoch+1} at {ckpt_path}, time {time.ctime()} ', file=open(logfile, 'a+'))  
        return "increased"
    else:
        return "decreased"

# Logs Saving File
version_adv = 'Adv-Train'
logfile  = f'Adv-Training/{version_adv}-{version}.log'

#%% log config
header1 = "** Experiment for PGD Attack Training **"
header2 = "** Num_Steps: {} | Alpha: {}/255 | Eps: {}/255 | Batch Size: {}**".format(num_steps, int(alpha * 255), int(eps * 255), int(batch_size))
# header3 = "** LR Config: Init: {} | Peak: {} | Warmup Epoch: {} | Rise: {} | Decay {} | train params {} | all params {} **".format([float(f'{sch.get_last_lr()[0]:.6f}') for sch in lr_schedulers], [float(f'{v:.6f}') for v in wp_lr], warmups, power, decay_rate, pytorch_train_params, pytorch_total_params)
maxlen  = len(header1)
# header1 = "{}{}{}".format('*'*((maxlen-len(header1))//2+1), header1, '*'*((maxlen-len(header1))//2+1) )
# header2 = "{}{}{}".format('*'*((maxlen-len(header2))//2+1), header2, '*'*((maxlen-len(header2))//2+1) )
# header3 = "{}{}{}".format('*'*((maxlen-len(header3))//2+1), header3, '*'*((maxlen-len(header3))//2+1) )
# maxlen  = max(len(header1), len(header2), len(header3))
print("\n\n\n{}\n{}\n{}\n{}\n\n".format("*"*maxlen, header1, header2, "*"*maxlen), file=open(logfile, 'a+'))
print("Experiment started ...\n   logging outputs to: ", logfile)

# WandB Initialization


# Save model inputs and hyperparameters in a wandb.config object
# config = run.config

# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "mAP_adv.compute_video_AP()['mAP']"},
    "parameters": {
        "lr_i": {"max": 0.01, "min": 0.0001},
        "lr_vt": {"max": 0.01, "min": 0.0001},
        "lr_ivt": {"max": 0.01, "min": 0.0001},
    },
}

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my_second_project")
def main():
    run = wandb.init(reinit=True)
    lr_i = wandb.config.lr_i
    lr_vt = wandb.config.lr_vt
    lr_ivt = wandb.config.lr_ivt

    wandb.log({"LRi": lr_i })
    wandb.log({"LRvt": lr_vt })
    wandb.log({"LRivt": lr_ivt })
    # config.learning_rate = 0.01

    # def objective(trial):
    # tool, verb-target, triplet
    # [0.01, 0.01, 0.01]

    # intermediate_results = {}

    # lr_i = trial.suggest_float('lr_tool', 0.00001, 0.01 , log=True)
    # lr_vt = trial.suggest_float('lr_verb_target', 0.00001, 0.01 , log=True)
    # lr_ivt = trial.suggest_float('lr_triplet', 0.00001, 0.01 , log=True)

    # lr_i = 7.252332280656785e-05
    # lr_vt = 0.0003426588582186996
    # lr_ivt = 0.0005371904315501744

    learning_rates = [lr_i , lr_vt , lr_ivt]

    print("Experiment LR: " , learning_rates )
    wp_lr           = [lr/power for lr in learning_rates]

    module_i        = list(set(model[1].parameters()) - set(model[1].encoder.cagam.parameters()) - set(model[1].encoder.bottleneck.parameters()) - set(model[1].decoder.parameters()))
    module_ivt      = list(set(model[1].encoder.bottleneck.parameters()).union(set(model[1].decoder.parameters())))
    module_vt       = model[1].encoder.cagam.parameters()

    optimizer_i     = torch.optim.SGD(module_i, lr=wp_lr[0], weight_decay=weight_decay)
    # scheduler_ia    = torch.optim.lr_scheduler.LinearLR(optimizer_i, start_factor=power, total_iters=warmups[0])
    scheduler_i    = torch.optim.lr_scheduler.ExponentialLR(optimizer_i, gamma=decay_rate)
    # scheduler_i     = torch.optim.lr_scheduler.SequentialLR(optimizer_i, schedulers=[scheduler_ia, scheduler_ib], milestones=[warmups[0]+1])

    optimizer_vt    = torch.optim.SGD(module_vt, lr=wp_lr[1], weight_decay=weight_decay)
    # scheduler_vta   = torch.optim.lr_scheduler.LinearLR(optimizer_vt, start_factor=power, total_iters=warmups[1])
    scheduler_vt   = torch.optim.lr_scheduler.ExponentialLR(optimizer_vt, gamma=decay_rate)
    # scheduler_vt    = torch.optim.lr_scheduler.SequentialLR(optimizer_vt, schedulers=[scheduler_vta, scheduler_vtb], milestones=[warmups[1]+1])

    optimizer_ivt   = torch.optim.SGD(module_ivt, lr=wp_lr[2], weight_decay=weight_decay)
    # scheduler_ivta  = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
    scheduler_ivt  = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
    # scheduler_ivt   = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb], milestones=[warmups[2]+1])

    lr_schedulers   = [scheduler_i, scheduler_vt, scheduler_ivt]
    optimizers      = [optimizer_i, optimizer_vt, optimizer_ivt]

    print("Total Epochs: " , epochs)

    for epoch in range(0,epochs):

        print("Current Epoch: " , epoch)
        
        try:
            
            # Train
            print("Traning | lr: {} | epoch {}".format([lr.get_last_lr() for lr in lr_schedulers], epoch), end=" | ", file=open(logfile, 'a+'))  
            
            train(train_dataloader, activation, loss_fn_i, loss_fn_v, loss_fn_t, loss_fn_ivt, optimizers, lr_schedulers, epoch)

            # val
            if epoch % val_interval == 0:
                print("HI")
                start = time.time()  
                # mAP.reset_global()
                # mAP_adv.reset_global()
                print("Evaluating @ epoch: ", epoch, file=open(logfile, 'a+'))
                # if epoch % 2 == 0:
                mAP.reset_global()
                mAP_adv.reset_global()
                test(val_dataloader, activation, final_eval=True)

                # intermediate_results[epoch] = {'mAP': mAP, 'mAP_adv': mAP_adv}
                # behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                behaviour = weight_mgt(mAP_adv.compute_video_AP()['mAP'], epoch=epoch)
                print("\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP Normal => ivt: [{:.5f}] ".format( (time.time() - start), mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP']), file=open(logfile, 'a+'))   
                print("\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP Adversarial => ivt: [{:.5f}] ".format( (time.time() - start), mAP_adv.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP']), file=open(logfile, 'a+'))   
            
            # intermediate_results[step] = {'metric1': metric1, 'metric2': metric2}
            # trial.report(mAP_adv.compute_video_AP()['mAP'], epoch)
            
            # Handle pruning based on the intermediate value.
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        except KeyboardInterrupt:
            print(f'>> Process cancelled by user at {time.ctime()}, ...', file=open(logfile, 'a+'))   
            sys.exit(1)
    # test_ckpt = ckpt_path
    print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-"*maxlen) , file=open(logfile, 'a+'))

wandb.agent(sweep_id, function=main, count=10)
# trial.set_user_attr('intermediate_results', intermediate_results)
# print("Hey Loss...! " , loss.item())

# return  mAP_adv.compute_video_AP()['mAP']

# 3. Create a study object and optimize the objective function.
# study = optuna.create_study(study_name="example-study", storage="sqlite:///example-study.db", direction='minimize' , load_if_exists=True)
# study.optimize(objective, n_trials=10)
# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

