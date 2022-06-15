import torch
import torch.nn as nn

class RDB(nn.Module):
    def __init__(self,in_c,out_c,slice_prop=4,layers=4):
        super(RDB,self).__init__()
        assert in_c==out_c
        assert layers>=2
        self.rate=1.
        self.refine=int(out_c*self.rate)
        self.layers=layers
        #self.head=nn.Conv2d(in_c,out_c,1,bias=False)
        layer_body=[]
        layer=[]
        layer.append(nn.Conv2d(out_c,out_c,3,padding=1,bias=False))
        layer.append(nn.PReLU(out_c,0.05))
        layer_body.append(nn.Sequential(*layer))
        
        for _ in range(layers-2):
            layer=[]
            layer.append(nn.Conv2d(self.refine,out_c,3,padding=1,bias=False))
            layer.append(nn.PReLU(out_c,0.05))
            layer_body.append(nn.Sequential(*layer))
            
        layer_body.append(nn.Conv2d(self.refine,out_c,3,padding=1,bias=False))
        self.body=nn.ModuleList(layer_body)
        
        self.bottle=nn.Conv2d(layers*out_c,out_c,1,bias=False)
        #channel attention
        layer=[]
        layer.append(nn.AdaptiveAvgPool2d(1))
        layer.append(nn.Conv2d(out_c,out_c//16,1,padding=0,bias=False))
        layer.append(nn.PReLU(out_c//16,0.05))
        layer.append(nn.Conv2d(out_c//16,out_c,1,padding=0,bias=False))
        layer.append(nn.Sigmoid())
        self.attention=nn.Sequential(*layer)
        #enhanced spatial attention
        self.esa_head=nn.Conv2d(out_c,out_c//4,1,bias=False)
        layer=[]
        layer.append(nn.Conv2d(out_c//4,out_c//4,3,stride=2,bias=False))
        layer.append(nn.MaxPool2d(7,stride=3))
        layer.append(nn.Conv2d(out_c//4,out_c//4,3,padding=1,bias=False))
        layer.append(nn.PReLU(out_c//4))
        layer.append(nn.Conv2d(out_c//4,out_c//4,3,padding=1,bias=False))
        layer.append(nn.PReLU(out_c//4))
        layer.append(nn.Conv2d(out_c//4,out_c//4,3,padding=1,bias=False))
        self.esa_body=nn.Sequential(*layer)
        self.esa_shortcut=nn.Conv2d(out_c//4,out_c//4,1,bias=False)
        layer=[]
        layer.append(nn.Conv2d(out_c//4,out_c,1,bias=False))
        layer.append(nn.Sigmoid())
        self.esa_tail=nn.Sequential(*layer)
    def forward(self,x):
        shortcut=x
        distilled=[]
        for i in range(self.layers):
            x=self.body[i](x)
            distilled.append(x)
            x=x[:,:self.refine]
        
        x=torch.cat(distilled,axis=1)
        x=self.bottle(x)
        attention=self.attention(x)

        #attention=self.esa_head(x)
        #att_shortcut=attention
        #attention=nn.Upsample((x.shape[2],x.shape[3]),mode='bilinear',align_corners=False)(self.esa_body(attention))
        #att_shortcut=self.esa_shortcut(att_shortcut)
        #attention=attention+att_shortcut
        #attention=self.esa_tail(attention)

        x=attention*x
        return x+shortcut

class backbone(nn.Module):
    def __init__(self,in_c,out_c,slice_prop_net=2,slice_prop_block=4,layers_net=2,layers_block=4,block_num=[1,12],block=RDB):
        super(backbone,self).__init__()
        assert layers_net>=2
        assert in_c==out_c
        self.slice_net=out_c//slice_prop_net
        self.layers=layers_net
        layer_body=[]
        layer=[]
        #stacked blocks
        if block_num[0]!=0:
            layer.append(block(in_c,out_c,slice_prop_block))
            for _ in range(block_num[0]-1):
                layer.append(block(out_c,out_c,slice_prop_block))
        layer_body.append(nn.Sequential(*layer))

        for i in range(layers_net-1):
            layer=[]
            if block_num[i+1]!=0:
                layer.append(block(out_c,out_c,slice_prop_block))
                for _ in range(block_num[i+1]-1):
                    layer.append(block(out_c,out_c,slice_prop_block))
            layer_body.append(nn.Sequential(*layer))
        
        self.body=nn.ModuleList(layer_body)
        layer=[]
        #feature distillation
        layer_ups=[]
        for i in range(layers_net-1):
            layer=[]
            layer.append(nn.PixelUnshuffle(2))
            layer.append(nn.Conv2d(2*4*out_c,out_c,1,bias=False))
            layer_ups.append(nn.Sequential(*layer))

        self.ups=nn.ModuleList(layer_ups)    
        #reconvery and reconstruction
        layer_ps=[]
        for i in range(layers_net-1):
            layer=[]
            layer.append(nn.Conv2d(out_c,4*out_c,1,bias=False))
            layer.append(nn.PixelShuffle(2))
            layer_ps.append(nn.Sequential(*layer))

        self.ps=nn.ModuleList(layer_ps)    
        #concat
        layer_bottle=[]
        for i in range(layers_net-1):
            layer=[]
            layer.append(nn.Conv2d(2*out_c,out_c,1,bias=False))
            layer_bottle.append(nn.Sequential(*layer))

        self.bottle=nn.ModuleList(layer_bottle)
        
        self.cat=nn.Conv2d(layers_net*out_c,out_c,1,bias=False)

    def forward(self,x):
        distilled=[]
        #channel separation
        distilled.append(x[:,:64])
        for i in range(self.layers-1):
            x=self.ups[i](x[:,64:])
            distilled.append(x)
        
        output=distilled[self.layers-1]
        for i in range(self.layers-1):
            output=self.body[self.layers-1-i](output)
            output=self.ps[self.layers-1-i-1](output)
            output=torch.cat([output,distilled[self.layers-1-i-1]],axis=1)
            output=self.bottle[self.layers-1-i-1](output)
        
        output=self.body[0](output)
        output=output+distilled[0]
        
        return output

class ResNet(nn.Module):
    def __init__(self,in_c,out_c,block=RFDB,layers_net=4,layers_block=2):
        super(ResNet,self).__init__()
        layer=[]
        for _ in range(layers_net):
            layer.append(block(in_c,out_c))
        
        self.body=nn.Sequential(*layer)
        
    def forward(self,x):
        res=x
        x=self.body(x)
        return x+res



class SCFDN(nn.Module):
    def __init__(self,scale=4,resblock=16):
        super(SCFDN, self).__init__()
        layer=[]
        layer.append(nn.Conv2d(3,64+128,3,padding=1,bias=False))
        layer.append(nn.PReLU(64+128))
        self.head=nn.Sequential(*layer)
        self.body=backbone(64,64)
        layer=[]
        for i in range(1):
            layer.append(nn.Conv2d(64,3*scale*scale,1,bias=False))
            layer.append(nn.PixelShuffle(scale))
        
        self.tail=nn.Sequential(*layer)
        
    def forward(self,x):
        x=self.head(x)
        x=self.body(x)
        result=self.tail(x)
        return result

if __name__=='__main__':
    from torchstat import stat
    with torch.no_grad():
        model=SCFDN()
        stat(model,(3,256,256))

