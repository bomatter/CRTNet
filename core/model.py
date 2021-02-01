import math

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_classes, num_decoder_layers=6, num_decoder_heads=8):
        super(Model, self).__init__()

        self.context_encoder = Encoder()
        self.target_encoder = Encoder()

        self.CONTEXT_IMAGE_SIZE = self.context_encoder.IMAGE_SIZE
        self.TARGET_IMAGE_SIZE = self.target_encoder.IMAGE_SIZE
        self.NUM_ENCODER_FEATURES = self.context_encoder.NUM_FEATURES

        assert(self.context_encoder.NUM_FEATURES == self.target_encoder.NUM_FEATURES), "Context and target encoder must extract the same number of features."
        
        self.tokenizer = Tokenizer()

        self.NUM_CONTEXT_TOKENS = self.tokenizer.NUM_CONTEXT_TOKENS
        self.NUM_TOKEN_FEATURES = self.NUM_ENCODER_FEATURES
        self.NUM_DECODER_HEADS = num_decoder_heads
        self.NUM_DECODER_LAYERS = num_decoder_layers 
        
        assert(self.NUM_TOKEN_FEATURES % self.NUM_DECODER_HEADS == 0), "NUM_TOKEN_FEATURES must be divisible by NUM_DECODER_HEADS."

        self.positional_encoding = PositionalEncoding(self.NUM_CONTEXT_TOKENS, self.NUM_TOKEN_FEATURES)

        self.decoder_layers = nn.TransformerDecoderLayer(self.NUM_TOKEN_FEATURES, nhead=self.NUM_DECODER_HEADS)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.NUM_DECODER_LAYERS)

        self.NUM_CLASSES = num_classes
        
        self.classifier = nn.Linear(self.NUM_TOKEN_FEATURES, self.NUM_CLASSES)

        self.initialize_weights()

    def forward(self, context_images, target_images, target_bbox):

        # Encoding of both streams
        # TODO: encoding could be done in parallel
        target_encoding = self.target_encoder(target_images)
        context_encoding = self.context_encoder(context_images)

        # TODO: Uncertainty gating for target

        # Tokenization and positional encoding
        context_encoding, target_encoding = self.tokenizer(context_encoding, target_encoding)
        context_encoding, target_encoding = self.positional_encoding(context_encoding, target_encoding, target_bbox)

        # Incorporation of context information using transformer decoder
        target_encoding = self.decoder(target_encoding, context_encoding)

        # Classification
        output = self.classifier(target_encoding.squeeze(0))

        return output

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = torchvision.models.densenet169(pretrained=False).features
        
        self.IMAGE_SIZE = (224, 224)
        self.NUM_FEATURES = 1664

    def forward(self, image):
        return self.encoder(image)


class Tokenizer(nn.Module):

    # TODO: make implementation more general so it works with different encodings and number of (context) tokens can be specified

    def __init__(self):
        super(Tokenizer, self).__init__()

        self.NUM_CONTEXT_TOKENS = 49
        self.NUM_TARGET_TOKENS = 1
        
    def forward(self, context_encoding, target_encoding):
        """
        Creates tokens from the encoded context and target.
        The token shapes are (NUM_CONTEXT_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES)
        and (NUM_TARGET_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES) respectively.
        """

        # one target token
        target_encoding = F.relu(target_encoding, inplace=True)
        target_encoding = F.adaptive_avg_pool2d(target_encoding, (1, 1))
        target_encoding = torch.flatten(target_encoding, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        target_encoding = torch.unsqueeze(target_encoding, 0) # output dimension: (NUM_TARGET_TOKENS=1, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # 49 context tokens
        context_encoding = F.relu(context_encoding, inplace=True)
        context_encoding = torch.flatten(context_encoding, 2, 3)
        context_encoding = context_encoding.permute(2, 0, 1) # output dimension: (NUM_CONTEXT_TOKENS=49, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        return context_encoding, target_encoding


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_context_tokens, num_token_features):
        super(PositionalEncoding, self).__init__()

        self.NUM_CONTEXT_TOKENS = num_context_tokens
        self.tokens_per_dim = int(math.sqrt(self.NUM_CONTEXT_TOKENS))
        self.positional_encoding = nn.Parameter(torch.zeros(num_context_tokens, 1, num_token_features))
        self.initialize_weights()

    def forward(self, context_tokens, target_tokens, target_bbox):
        context_tokens = context_tokens + self.positional_encoding
        target_tokens = target_tokens + torch.index_select(self.positional_encoding, 0, self.bbox2token(target_bbox)).permute(1,0,2)

        return context_tokens, target_tokens

    def bbox2token(self, bbox):
        """
        Maps relative bbox coordinates to the corresponding token ids (e.g., 0 for the token in the top left).

        Arguments:
            bbox: Tensor of dim (batch_size, 4) where a row corresponds to relative coordinates
                  in the form [xmin, ymin, w, h] (e.g., [0.1, 0.3, 0.2, 0.2]).
        """
        
        token_ids = ((torch.ceil((bbox[:,0] + bbox[:,2]/2) * self.tokens_per_dim) - 1) + (torch.ceil((bbox[:,1] + bbox[:,3]/2) * self.tokens_per_dim) - 1) * self.tokens_per_dim).long()

        return token_ids
        
    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)