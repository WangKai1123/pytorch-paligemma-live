
from typing import Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_hidden_layer=12,
                 num_attemtion_heads=12,
                 num_channel=3,
                 image_size=224,
                 patch_size=16,
                 layer_norm_eps=1e-6,
                 attention_dropout=0.0,
                 num_image_tokens: int =None,
                 **kwargs
                 ):
                super().__init__()
                self.hidden_size=hidden_size,
                self.intermediate_size=intermediate_size,
                self.num_hidden_layer=num_hidden_layer,
                self.num_attemtion_heads=num_attemtion_heads
                self.num_channel=num_channel,
                self.image_size=image_size,
                self.patch_size=patch_size,
                self.layer_norm_eps=layer_norm_eps,
                self.attention_dropout=attention_dropout,
                self.num_image_tokens=num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
        def __init__(self, config: SiglipVisionConfig):
                super().__init__()
                self.config = config
                self.embed_dim = config.hidden_size
                self.image_size = config.image_size
                self.patch_size = config.patch_size

                self.patch_embedding =  nn.Conv2d(
                        in_channels=config.num_channel,
                        out_channels=self.embed_dim,
                        kernel_size=self.patch_size,
                        stride=self.patch_size,
                        padding="valid",
                )
                self.num_patches = (self.image_size // self.patch_size) **2
                self.num_positions = self.num_patches
                self.position_embedding = nn.Embedding(self.num_patches,self.embed_dim)
                self.register_buffer(
                        "position_ids",
                        torch.arange(self.num_positions).expand((1,-1)),
                        persistent=False
                )
                



class SiglipVisionTransformer(nn.Module):
        def __init__(self, config: SiglipVisionConfig):
                super().__init__()
                self.config = config
                embed_dim = config.hidden_size

                self.embeddings = SiglipVisionEmbedding(config)
                self.encoder = SiglipEncoder(config)
                self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

        def forward(self,pixel_values:torch.Tensor) -> torch.Tensor:
             #pixel_values:[Batch_size, Channels, Height, Width] ->[Batch_Size, Num_Patches,Embed_dim]
             hidden_states = self.embeddings(pixel_values)
             last_hidden_state = self.encoder(input_embeds=hidden_states)
             last_hidden_state = self.post_layernorm(last_hidden_state)
             return last_hidden_state




class SiglipVisionModel(nn.Module):
        
        def __init__(self, config: SiglipVisionConfig):
                super().__init__()
                #这个视觉模型游config文件和下面的视觉文件组成
                self.config = config
                self.vision_model = SiglipVisionTransformer(config)

        def forward(self,pix_values) ->Tuple:
                #把图像进行token化
                #[Batch_size Channels  Height Width] ->[Batch_Size, Num_Patches, Embed_dim]
                return self.vision_model(pix_values=pix_values)