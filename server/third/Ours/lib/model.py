"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class BaseNet(nn.Module):
	def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
		super(BaseNet, self).__init__()
		self.inchan = inchan
		self.curchan = inchan
		self.dilation = dilation
		self.bn = bn
		self.bn_affine = bn_affine
		self.ops = nn.ModuleList([])

	def _make_bn(self, outd):
		return nn.BatchNorm2d(outd, affine=self.bn_affine, momentum=0.1)

	def MakeBlk(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, ):
		d = self.dilation * dilation

		conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
		t = nn.ModuleList([])
		t.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
		if bn and self.bn: t.append(self._make_bn(outd))
		if relu: t.append(nn.ReLU(inplace=True))
		blk = nn.Sequential(*t)
		self.curchan = outd
		self.dilation *= stride

		return blk


class Adapter(BaseNet):
	def __init__(self, mchan=4, **kw):
		super(Adapter, self).__init__()
		t = BaseNet()
		tt = nn.ModuleList([])
		ops = nn.ModuleList([])
		ops.append(t.MakeBlk(8 * mchan))
		ops.append(t.MakeBlk(8 * mchan))
		ops.append(t.MakeBlk(16 * mchan, stride=2))
		ops.append(t.MakeBlk(16 * mchan))
		ops.append(t.MakeBlk(32 * mchan, stride=2))
		ops.append(t.MakeBlk(32 * mchan))
		self.ops = ops
		self.RLNs = tt

	def forward(self, x):
		for i, layer in enumerate(self.ops):
			if i % 2 == 1:
				x = layer(x) + x
			else:
				x = layer(x)
		return x


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.relu = torch.nn.ReLU(inplace=True)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

		# Shared Encoder.
		self.conv1a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.conv1b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.conv2a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.conv2b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.conv3a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.conv4a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.conv4b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.block_fusion = nn.Sequential(
			nn.Conv2d(128, 128, 3, padding=1, stride=1, dilation=1, bias=False),
			nn.BatchNorm2d(128, affine=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1, stride=1, dilation=1, bias=False),
			nn.BatchNorm2d(128, affine=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 1, padding=0)
		)

	def forward(self, x):
		x1 = self.relu(self.conv1a(x))
		x1 = self.relu(self.conv1b(x1))
		x1 = self.pool(x1)
		x1 = self.relu(self.conv2a(x1))
		x1 = self.relu(self.conv2b(x1))
		x1 = self.pool(x1)

		x2 = self.relu(self.conv3a(x1))
		x2 = self.relu(self.conv3b(x2))
		x2 = self.pool(x2)
		x2 = self.relu(self.conv4a(x2))
		x2 = self.relu(self.conv4b(x2))

		# pyramid fusion
		x1 = F.interpolate(x1, (x.shape[-2], x.shape[-1]), mode='bilinear')
		x2 = F.interpolate(x2, (x.shape[-2], x.shape[-1]), mode='bilinear')
		feats = self.block_fusion(x + x1 + x2)
		return feats


class Encoder_Fine(nn.Module):
    def __init__(self, dim=128, mchan=4, relu22=False, dilation=4,**kw ):
        super(Encoder_Fine,self).__init__()
        t = BaseNet(inchan=32*mchan, dilation=dilation)
        ops=nn.ModuleList([])
        ops.append(t.MakeBlk(32*mchan, k=3, stride=2, relu=False))
        ops.append(t.MakeBlk(32*mchan, k=3, stride=2, relu=False))
        ops.append(t.MakeBlk(dim, k=3, stride=2, bn=False, relu=False))
        self.out_dim = dim
        self.ops = ops

    def forward(self,x):
        for i in range(len(self.ops)):
            if i%2==1:
                x = self.ops[i](x)+x
            else:
                x = self.ops[i](x)
        return x
	

class ConditionalEstimator(nn.Module):
	def __init__(self) -> None:
		super(ConditionalEstimator, self).__init__()
		self.dropout = nn.Dropout(0.1)
		self.preconv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
									 nn.BatchNorm2d(64, affine=False),
									 nn.ReLU(),
									 nn.Conv2d(64, 32, kernel_size=3, dilation=2, padding=2, bias=False),
									 nn.BatchNorm2d(32, affine=False),
									 nn.ReLU(),
									 nn.Conv2d(32, 16, kernel_size=3, dilation=4, padding=4, bias=False),
									 nn.BatchNorm2d(16, affine=False),
									 nn.ReLU()
									 )
		self.bn1 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
								 nn.ReLU(),
								 nn.InstanceNorm2d(16, affine=False),
								 nn.ReLU())
		self.bn2 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
								 nn.ReLU(),
								 nn.InstanceNorm2d(16, affine=False),
								 nn.ReLU())
		self.bn3 = nn.Sequential(nn.BatchNorm2d(16, affine=False),
								 nn.ReLU(),
								 nn.InstanceNorm2d(16, affine=False),
								 nn.ReLU())
		self.pool1 = nn.AvgPool2d(3, stride=1, padding=1)
		self.pool2 = nn.AvgPool2d(3, stride=1, padding=1)
		self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
		self.layer1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
									nn.BatchNorm2d(16, affine=False),
									nn.ReLU(),
									)
		self.layer2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
									nn.BatchNorm2d(16, affine=False),
									nn.ReLU(),
									)
		self.layer3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False),
									nn.BatchNorm2d(16, affine=False),
									nn.ReLU())
		self.postconv = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1, bias=False))

	def LN(self, x, conv, pool, bn):
		x = conv(x)
		x = x.exp().clamp(max=1e4)
		x = x / (pool(x) + 1e-5)
		x = bn(x)
		return x

	def forward(self, x):
		x = self.dropout(x)
		x = self.preconv(x)
		x = self.LN(x, self.layer1, self.pool1, self.bn1)
		x = self.LN(x, self.layer2, self.pool2, self.bn2)
		x = self.LN(x, self.layer3, self.pool3, self.bn3)
		x = self.postconv(x)
		x = F.softmax(x, dim=1)[:, 0].unsqueeze(1)
		return x


class Superhead(nn.Module):
	def __init__(self) -> None:
		super(Superhead, self).__init__()
		self.PriorEstimator = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1))

		self.ConditionalEstimator = ConditionalEstimator()

	def p_x(self, x):
		x = self.PriorEstimator(x)
		p_x = F.softplus(x)
		p_x = p_x / (1 + p_x)
		# p_x = x/(1+x)
		return p_x

	def forward(self, x):
		p_c = self.ConditionalEstimator(x)
		p_x = self.p_x(x)
		# p_y = F.softplus(x1+x2)
		# p_y = p_y/(1+p_y)
		# p_x = self.p_x(x)
		p_y = (p_x * p_c)
		return p_y
	

class Fine_matcher(nn.Module):
	def __init__(self):
		super().__init__()
		self.bb = nn.Sequential(
								nn.Conv2d(131, 128, 3, 1, 1),
								nn.BatchNorm2d(128, affine=False),
								nn.ReLU(inplace = True),
								nn.Conv2d(128, 128, 3, 1, 1),
								nn.BatchNorm2d(128, affine=False),
								nn.ReLU(inplace = True),
								nn.Conv2d(128, 128, 3, 1, 1),
		)

	def forward(self, patch1, patch2):
		# patch1 = self.bb(patch1)
		mask = (patch1 * patch2).sum(dim=1, keepdim=True)
		return mask
	

class XFeatModel(nn.Module):
	def __init__(self, fine_model='add'):
		super().__init__()
		self.ada1 = Adapter()
		self.ada2 = Adapter()
		self.enc = Encoder()
		self.det = Superhead()

		self.enc_fine = Encoder_Fine()
		self.fine_matcher = Fine_matcher()

		self.fine_model = fine_model

	def forward1(self, x):
		c_feat, f_feat, score = [], [], []
		if self.fine_model == 'add':
			x = self.ada1(x)
			c_feat = self.enc(x)
			score = self.det(c_feat.pow(2))
			f_feat = self.enc_fine(c_feat)
		else:
			c_feat = self.ada1(x)
			f_feat = self.enc(c_feat)
			score = self.det(c_feat.pow(2))
		return c_feat, f_feat, score

	def forward2(self, x):
		c_feat, f_feat, score = [], [], []
		if self.fine_model == 'add':
			x = self.ada2(x)
			c_feat = self.enc(x)
			score = self.det(c_feat.pow(2))
			f_feat = self.enc_fine(c_feat)
		else:
			c_feat = self.ada2(x)
			f_feat = self.enc(c_feat)
			score = self.det(c_feat.pow(2))
		return c_feat, f_feat, score

	def forward(self, im1, im2):
		c_feat1, f_feat1, score1 = self.forward1(im1)
		c_feat2, f_feat2, score2 = self.forward2(im2)
		return c_feat1, f_feat1, score1, c_feat2, f_feat2, score2
