import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.context_encoder = Encoder()
        self.target_encoder = Encoder()

        self.CONTEXT_IMAGE_SIZE = self.context_encoder.IMAGE_SIZE
        self.TARGET_IMAGE_SIZE = self.target_encoder.IMAGE_SIZE

        assert(self.context_encoder.NUM_FEATURES == self.target_encoder.NUM_FEATURES), "Context and target encoder must extract the same number of features."
        
        self.NUM_ENCODER_FEATURES = self.context_encoder.NUM_FEATURES
        
        self.tokenizer = Tokenizer()

        self.NUM_TOKEN_FEATURES = self.NUM_ENCODER_FEATURES
        self.NUM_DECODER_HEADS = 8 # TODO: specify through config, check requirement that NUM_TOKEN_FEATURES must be divisible by NUM_DECODER_HEADS
        self.NUM_DECODER_LAYERS = 6 # TODO: specify through config

        self.decoder_layers = nn.TransformerDecoderLayer(self.NUM_TOKEN_FEATURES, nhead=self.NUM_DECODER_HEADS)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.NUM_DECODER_LAYERS)

        self.NUM_CLASSES = num_classes
        
        self.classifier = nn.Linear(self.NUM_TOKEN_FEATURES, self.NUM_CLASSES)

        self.initialize_weights()

    def forward(self, context_images, target_images):

        # Encoding of both streams
        # TODO: encoding could be done in parallel
        target_encoding = self.target_encoder(target_images)
        context_encoding = self.context_encoder(context_images)

        # TODO: Uncertainty gating for target        

        # Tokenization
        # TODO: positional encoding
        context_encoding, target_encoding = self.tokenizer(context_encoding, target_encoding)

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

        self.encoder = torchvision.models.densenet169(pretrained=True).features
        
        self.IMAGE_SIZE = (224, 224)
        self.NUM_FEATURES = 1664

    def forward(self, input):
        return self.encoder(input)


class Tokenizer(nn.Module):

    def __init__(self, positional_encoding=None):
        super(Tokenizer, self).__init__()
        
        #self.positional_encoding = positional_encoding

    def forward(self, context_encoding, target_encoding):
        """
        Creates tokens from the encoded context and target.
        The token shapes are (NUM_CONTEXT_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES)
        and (NUM_TARGET_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES) respectively.
        """
        # TODO: make implementation more general so it works with different encodings and number of (context) tokens can be specified

        # one target token
        target_encoding = F.relu(target_encoding, inplace=True)
        target_encoding = F.adaptive_avg_pool2d(target_encoding, (1, 1))
        target_encoding = torch.flatten(target_encoding, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        target_encoding = torch.unsqueeze(target_encoding, 0) # output dimension: (NUM_TARGET_TOKENS=1, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # 49 context tokens
        context_encoding = F.relu(context_encoding, inplace=True)
        context_encoding = torch.flatten(context_encoding, 2, 3)
        context_encoding = context_encoding.permute(2, 0, 1) # output dimension: (NUM_CONTEXT_TOKENS=49, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # TODO: positional encoding

        return context_encoding, target_encoding

class PositionalEncoding:
    pass