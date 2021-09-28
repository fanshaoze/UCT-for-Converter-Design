import torch
import torch.nn as nn 
from model.Layers import EncoderLayer, DecoderLayer
from model.Embed import Embedder, PositionalEncoder
from model.Sublayers import FeedForward, MultiHeadAttention, Norm
import copy
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

Constants_PAD = 0

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        # self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)
        # self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        # self.dropout= nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # x = self.embed(src)
        # x = self.pe(x)
        # x = src
        for i in range(self.N_layers):
            x = self.layers[i](x,mask)
        return self.norm(x)

    # def forward(self, image1, image2):
    #     image1 = self.norm_1(image1)
    #     image2 = self.norm_2(image2)
    #     x = self.dropout(self.attn(image1,image2,image2))
    #     for i in range(self.N_layers):
    #         x = self.layers[i](x)
    #     return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, src_mask=None, trg_mask=trg_mask)
        return self.norm(x)




class PathEmbedding(nn.Module):
    def __init__(self, d_model, attribute_vocab_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super().__init__()
        self.embed = Embedder(attribute_vocab_size, d_model)
        #self.norm = Norm(d_model)

    def forward(self, attribute):
        attribute = self.embed(attribute)
        return attribute
        #return self.norm(attribute)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, cnn_model_name, \
        joint_encoding_function, attribute_vocab_size=1000, cnn_pretrained_model=None, add_attribute=False):
        super().__init__()
        # self.add_attribute = add_attribute
        # self.cnn1 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)
        # self.cnn2 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)
        # self.bn = nn.BatchNorm1d(d_model, momentum=0.01)

        # if self.add_attribute:
        self.embedding = Embedder(vocab_size,d_model)
            # self.attribute_embedding2 = Attribute_Embedding(d_model, attribute_vocab_size)
        # self.joint_encoding = Joint_Encoding(joint_encoding_function)
        self.encoder = Encoder(d_model, N, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, 1)

    # def get_trainable_parameters(self):
    #     if not self.add_attribute:
    #         return list(self.cnn1.parameters()) \
    #             + list(self.cnn2.get_trainable_parameters() \
    #             +  list(self.encoder.parameters()) \
    #             + list(self.decoder.parameters()) \
    #             + list(self.out.parameters())
    #     else:
    #         return list(self.cnn1.parameters()) \
    #                 + list(self.cnn2.parameters()) \
    #                 +  list(self.encoder.parameters()) \
    #                 + list(self.decoder.parameters()) \
    #                 + list(self.out.parameters()) \
    #                 + list(self.attribute_embedding1.parameters()) \
    #                 + list(self.attribute_embedding2.parameters()) \
    #                 # + list(self.bn.parameters())

    # def get_parameters_to_initial(self):
    #     return list(self.encoder.parameters()) \
    #             + list(self.decoder.parameters()) \
    #             + list(self.out.parameters()) \
    #             + list(self.attribute_embedding1.parameters()) \
    #             + list(self.attribute_embedding2.parameters())


    def forward(self, x, padding_mask):
        #image1, image2 = image2, image1
        #padding_mask = (x != Constants_PAD)
        #print(x.shape, padding_mask.shape)
        padding_mask = padding_mask.unsqueeze(2)
        x = self.embedding(x)
        x = self.encoder(x,padding_mask)
        x_1 = torch.mean(x, 1)
        x = self.out(x_1)
        #x = torch.mean(x, 1)
        return x_1,x

#TODO
class SVGPTransformerModel(nn.Module):
    def __init__(self, args, ):
        super().__init__()
        self.gp = get_gp_model
        self.transformer = transformer
    def forward(self, x,padding_mask):
        x = self.transformer(x,padding_mask)
        x = self.gp(x)

        return x

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# TODO
# def get_gp_model(opt, load_weights=False):

def get_model(opt=None, cuda=None, pretrained_model=None, load_weights=False):
    if opt is not None:
        cuda = opt.cuda
        pretrained_model = opt.pretrained_model

    if load_weights:
        checkpoint = torch.load(pretrained_model + '.chkpt')

        model_opt = checkpoint['settings']

        model = Transformer(model_opt.vocab_size, model_opt.d_model, \
            model_opt.n_layers, model_opt.n_heads, model_opt.dropout, \
            model_opt.cnn_name, model_opt.joint_enc_func, \
            model_opt.attribute_vocab_size, model_opt.cnn_pretrained_model, model_opt.add_attribute,
            )

        model.load_state_dict(checkpoint['model'])
        
        print('[Info] Trained model state loaded from: ', pretrained_model)

    else:
        assert opt.d_model % opt.n_heads == 0

        assert opt.dropout < 1

        model = Transformer(opt.vocab_size, opt.d_model, opt.n_layers, opt.n_heads, opt.dropout, \
            opt.cnn_name, opt.joint_enc_func, opt.attribute_vocab_size, opt.cnn_pretrained_model, \
            opt.add_attribute)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    return model

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask

def create_masks(trg):
    # src_mask = (src != Constants_PAD.unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != Constants_PAD).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size).to(trg_mask.device)

        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None

    return trg_mask



    
