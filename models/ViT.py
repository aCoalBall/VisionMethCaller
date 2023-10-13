from torch import nn
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoConfig, AutoModel

class ClassificationHead(nn.Module):
    def __init__(self, n_inputs, n_classes, head_dropout) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_inputs, n_classes)
    
    def forward(self, x):
        """
        x: [bs x n_inputs]
        output: [bs x n_classes]
        """
        x = self.flatten(x)         # x: bs x n_inputs #197x768 for vit-base; 197x192 for vit-tiny
        x = self.dropout(x)
        y = self.linear(x)          #y : bs x n_classes
        return y

class ViT(nn.Module):
    def __init__(self, n_classes:int, head_dropout:int) -> None:
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.head = ClassificationHead(n_inputs=197 * 768, n_classes=n_classes, head_dropout=head_dropout)
    
    def forward(self, x):
        """
        xS: tensor [bs x 224 x 224 x 3]
        """   
        x = self.vit(x)                   # x: bs x 197 x 768                              
        x = self.head(x.last_hidden_state)   # x: bs x n_classes
        return x
    
class ViT_tiny(nn.Module):
    def __init__(self, n_classes:int, head_dropout:int) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        self.vit =  AutoModel.from_config(config)
        self.head = ClassificationHead(n_inputs=197 * 192, n_classes=n_classes, head_dropout=head_dropout)
    
    def forward(self, x):
        """
        xS: tensor [bs x 224 x 224 x 3]
        """   
        x = self.vit(x)                   # x: bs x 197 x 192                             
        x = self.head(x.last_hidden_state)   # x: bs x n_classes
        return x   
    
class Swin(nn.Module):
    def __init__(self, n_classes:int, head_dropout:int) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.vit =  AutoModel.from_config(config)
        self.head = ClassificationHead(n_inputs=49 * 768, n_classes=n_classes, head_dropout=head_dropout)
    
    def forward(self, x):
        """
        xS: tensor [bs x 224 x 224 x 3]
        """   
        x = self.vit(x)                   # x: bs x 197 x 192                             
        x = self.head(x.last_hidden_state)   # x: bs x n_classes
        return x   