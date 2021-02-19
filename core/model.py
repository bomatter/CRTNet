import math

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_classes, num_decoder_layers=6, num_decoder_heads=8, uncertainty_threshold=0, extended_output=False, gpu_streams=True):
        """
        BigPictureNet Model

        Args:
            num_classes (int): Number of classes.
            num_decoder_layers (int, optional): Defaults to 6.
            num_decoder_heads (int, optional): Defaults to 8.
            uncertainty_threshold (int, optional): Used for the uncertainty gating mechanism. If the prediction uncertainty exceeds the uncertainty_threshold, context information is incorporated. Defaults to 0.
            extended_output (bool, optional): Can be enabled to return predictions from both branches and uncertainty value (as during training) when the model is in evaluation mode.
            gpu_streams (bool, optional): If set to True and GPUs are available, multiple gpu streams may be used to parallelize encoding. Defaults to True.
        """        

        super(Model, self).__init__()

        self.NUM_CLASSES = num_classes

        self.context_encoder = Encoder()
        self.target_encoder = Encoder()

        self.CONTEXT_IMAGE_SIZE = self.context_encoder.IMAGE_SIZE
        self.TARGET_IMAGE_SIZE = self.target_encoder.IMAGE_SIZE
        self.NUM_ENCODER_FEATURES = self.context_encoder.NUM_FEATURES

        assert(self.context_encoder.NUM_FEATURES == self.target_encoder.NUM_FEATURES), "Context and target encoder must extract the same number of features."
        
        self.uncertainty_gate = UncertaintyGate(self.NUM_ENCODER_FEATURES, self.NUM_CLASSES)
        
        self.UNCERTAINTY_THRESHOLD = uncertainty_threshold

        self.tokenizer = Tokenizer()

        self.NUM_CONTEXT_TOKENS = self.tokenizer.NUM_CONTEXT_TOKENS
        self.NUM_TOKEN_FEATURES = self.NUM_ENCODER_FEATURES
        self.NUM_DECODER_HEADS = num_decoder_heads
        self.NUM_DECODER_LAYERS = num_decoder_layers 
        
        assert(self.NUM_TOKEN_FEATURES % self.NUM_DECODER_HEADS == 0), "NUM_TOKEN_FEATURES must be divisible by NUM_DECODER_HEADS."

        self.positional_encoding = PositionalEncoding(self.NUM_CONTEXT_TOKENS, self.NUM_TOKEN_FEATURES)

        self.decoder_layers = nn.TransformerDecoderLayer(self.NUM_TOKEN_FEATURES, nhead=self.NUM_DECODER_HEADS)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.NUM_DECODER_LAYERS)
 
        self.classifier = nn.Linear(self.NUM_TOKEN_FEATURES, self.NUM_CLASSES)

        self.initialize_weights()

        # cuda streams for parallel encoding
        self.gpu_streams = True if gpu_streams and torch.cuda.is_available() else False
        self.target_stream = torch.cuda.Stream(priority=-1) if self.gpu_streams else None # set target stream as high priority because context encoding may not be necessary due to uncertainty gating
        self.context_stream = torch.cuda.Stream() if self.gpu_streams else None

        self.extended_output = extended_output

    def forward(self, context_images, target_images, target_bbox):

        # Encoding of both streams
        if self.gpu_streams:
            torch.cuda.synchronize()
            
        with torch.cuda.stream(self.target_stream):
            target_encoding = self.target_encoder(target_images)
            
            # Uncertainty gating for target
            prediction, uncertainty = self.uncertainty_gate(target_encoding) # predictions and associated confidence metrics
            
            # During inference, return prediction if prediction uncertainty is below the specified uncertainty threshold.
            # Note: The current implementation makes the gating decision on a per-batch basis. We expect/recommend that a batch size of 1 is used for inference.
            if not self.training and not self.extended_output and torch.all(uncertainty < self.UNCERTAINTY_THRESHOLD).item():
                return prediction
            
        with torch.cuda.stream(self.context_stream):
            context_encoding = self.context_encoder(context_images)
        
        if self.gpu_streams:
            torch.cuda.synchronize()

        # Tokenization and positional encoding
        context_encoding, target_encoding = self.tokenizer(context_encoding, target_encoding)
        context_encoding, target_encoding = self.positional_encoding(context_encoding, target_encoding, target_bbox)

        # Incorporation of context information using transformer decoder
        target_encoding = self.decoder(target_encoding, context_encoding)

        # Classification
        output = self.classifier(target_encoding.squeeze(0))

        if self.training or self.extended_output:
            return prediction, output, uncertainty # return both predictions (from uncertainty gating branch and main branch) and uncertainty
        else:
            return output

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def unfreeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = True


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = torchvision.models.densenet169(pretrained=True).features
        
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
        target_encoding = F.relu(target_encoding)
        target_encoding = F.adaptive_avg_pool2d(target_encoding, (1, 1))
        target_encoding = torch.flatten(target_encoding, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        target_encoding = torch.unsqueeze(target_encoding, 0) # output dimension: (NUM_TARGET_TOKENS=1, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # 49 context tokens
        context_encoding = F.relu(context_encoding)
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


class UncertaintyGate(nn.Module):

    def __init__(self, num_features, num_classes):
        super(UncertaintyGate, self).__init__()
        self.target_classifier = nn.Linear(num_features, num_classes)
        self.initialize_weights()

    def forward(self, input_features):
        # TODO: could add a few layers here
        
        # flatten featuremap out
        input_features = F.relu(input_features)
        input_features = F.adaptive_avg_pool2d(input_features, (1, 1))
        input_features = torch.flatten(input_features, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        
        predictions = self.target_classifier(input_features)
        entropy = -1 * torch.sum(F.softmax(predictions.detach(), dim=1) * F.log_softmax(predictions.detach(), dim=1), dim=1) # entropy as metric for uncertainty

        return predictions, entropy

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)