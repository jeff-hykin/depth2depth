// Vendored from candle-transformers, patched: rectangular inputs + metric head (sigmoid * max_depth).

use std::sync::Arc;

use candle::{Module, Result, Tensor};
use candle_nn::ops::Identity;
use candle_nn::{
    conv2d, conv2d_no_bias, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2dConfig,
    VarBuilder,
};

use crate::dinov2::DinoVisionTransformer;

pub struct DepthAnythingV2Config {
    out_channel_sizes: [usize; 4],
    in_channel_size: usize,
    num_features: usize,
    layer_ids_vits: Vec<usize>,
    input_h: usize,
    input_w: usize,
    patch_h: usize,
    patch_w: usize,
    max_depth: f64,
}

impl DepthAnythingV2Config {
    pub fn vit_small_metric(input_h: usize, input_w: usize, max_depth: f64) -> Self {
        Self {
            out_channel_sizes: [48, 96, 192, 384],
            in_channel_size: 384,
            num_features: 64,
            layer_ids_vits: vec![2, 5, 8, 11],
            input_h,
            input_w,
            patch_h: input_h / 14,
            patch_w: input_w / 14,
            max_depth,
        }
    }
}

pub struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    pub fn new(conf: &DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(conf.num_features, conf.num_features, 3, conv_cfg, vb.pp("conv1"))?;
        let conv2 = conv2d(conf.num_features, conf.num_features, 3, conv_cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv1.forward(&xs.relu()?)?;
        let out = self.conv2.forward(&out.relu()?)?;
        out + xs
    }
}

pub struct FeatureFusionBlock {
    res_conv_unit1: ResidualConvUnit,
    res_conv_unit2: ResidualConvUnit,
    output_conv: Conv2d,
    target_h: usize,
    target_w: usize,
}

impl FeatureFusionBlock {
    pub fn new(
        conf: &DepthAnythingV2Config,
        target_h: usize,
        target_w: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let output_conv = conv2d(
            conf.num_features,
            conf.num_features,
            1,
            Default::default(),
            vb.pp("out_conv"),
        )?;
        let res_conv_unit1 = ResidualConvUnit::new(conf, vb.pp("resConfUnit1"))?;
        let res_conv_unit2 = ResidualConvUnit::new(conf, vb.pp("resConfUnit2"))?;
        Ok(Self {
            res_conv_unit1,
            res_conv_unit2,
            output_conv,
            target_h,
            target_w,
        })
    }
}

impl Module for FeatureFusionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.res_conv_unit2.forward(xs)?;
        let out = out.interpolate2d(self.target_h, self.target_w)?;
        self.output_conv.forward(&out)
    }
}

pub struct Scratch {
    layer1_rn: Conv2d,
    layer2_rn: Conv2d,
    layer3_rn: Conv2d,
    layer4_rn: Conv2d,
    refine_net1: FeatureFusionBlock,
    refine_net2: FeatureFusionBlock,
    refine_net3: FeatureFusionBlock,
    refine_net4: FeatureFusionBlock,
    output_conv1: Conv2d,
    output_conv2: OutputConv2,
}

/// conv - relu - conv - sigmoid; a concrete type because candle_nn::Sequential
/// boxes `dyn Module` without Send + Sync.
pub struct OutputConv2 {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Module for OutputConv2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?.relu()?;
        candle_nn::ops::sigmoid(&self.conv2.forward(&xs)?)
    }
}

impl Scratch {
    pub fn new(conf: &DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let layer1_rn = conv2d_no_bias(
            conf.out_channel_sizes[0],
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("layer1_rn"),
        )?;
        let layer2_rn = conv2d_no_bias(
            conf.out_channel_sizes[1],
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("layer2_rn"),
        )?;
        let layer3_rn = conv2d_no_bias(
            conf.out_channel_sizes[2],
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("layer3_rn"),
        )?;
        let layer4_rn = conv2d_no_bias(
            conf.out_channel_sizes[3],
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("layer4_rn"),
        )?;

        let (ph, pw) = (conf.patch_h, conf.patch_w);
        let refine_net1 = FeatureFusionBlock::new(conf, ph * 8, pw * 8, vb.pp("refinenet1"))?;
        let refine_net2 = FeatureFusionBlock::new(conf, ph * 4, pw * 4, vb.pp("refinenet2"))?;
        let refine_net3 = FeatureFusionBlock::new(conf, ph * 2, pw * 2, vb.pp("refinenet3"))?;
        let refine_net4 = FeatureFusionBlock::new(conf, ph, pw, vb.pp("refinenet4"))?;

        let output_conv1 = conv2d(
            conf.num_features,
            conf.num_features / 2,
            3,
            conv_cfg,
            vb.pp("output_conv1"),
        )?;

        let output_conv2 = OutputConv2 {
            conv1: conv2d(
                conf.num_features / 2,
                32,
                3,
                conv_cfg,
                vb.pp("output_conv2").pp("0"),
            )?,
            conv2: conv2d(32, 1, 1, Default::default(), vb.pp("output_conv2").pp("2"))?,
        };

        Ok(Self {
            layer1_rn,
            layer2_rn,
            layer3_rn,
            layer4_rn,
            refine_net1,
            refine_net2,
            refine_net3,
            refine_net4,
            output_conv1,
            output_conv2,
        })
    }
}

const NUM_CHANNELS: usize = 4;

pub struct DPTHead {
    projections: Vec<Conv2d>,
    resize_layers: Vec<Box<dyn Module + Send + Sync>>,
    scratch: Scratch,
    input_h: usize,
    input_w: usize,
    patch_h: usize,
    patch_w: usize,
}

impl DPTHead {
    pub fn new(conf: &DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        let mut projections: Vec<Conv2d> = Vec::with_capacity(NUM_CHANNELS);
        for (conv_index, out_channel_size) in conf.out_channel_sizes.iter().enumerate() {
            projections.push(conv2d(
                conf.in_channel_size,
                *out_channel_size,
                1,
                Default::default(),
                vb.pp("projects").pp(conv_index.to_string()),
            )?);
        }

        let resize_layers: Vec<Box<dyn Module + Send + Sync>> = vec![
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[0],
                conf.out_channel_sizes[0],
                4,
                ConvTranspose2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("0"),
            )?),
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[1],
                conf.out_channel_sizes[1],
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("1"),
            )?),
            Box::new(Identity::new()),
            Box::new(conv2d(
                conf.out_channel_sizes[3],
                conf.out_channel_sizes[3],
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("3"),
            )?),
        ];

        let scratch = Scratch::new(conf, vb.pp("scratch"))?;

        Ok(Self {
            projections,
            resize_layers,
            scratch,
            input_h: conf.input_h,
            input_w: conf.input_w,
            patch_h: conf.patch_h,
            patch_w: conf.patch_w,
        })
    }
}

impl Module for DPTHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(NUM_CHANNELS);
        for i in 0..NUM_CHANNELS {
            let x = xs.get(i)?;
            let x_dims = x.dims();
            let x = x.permute((0, 2, 1))?.reshape((
                x_dims[0],
                x_dims[x_dims.len() - 1],
                self.patch_h,
                self.patch_w,
            ))?;
            let x = self.projections[i].forward(&x)?;
            let x = self.resize_layers[i].forward(&x)?;
            out.push(x);
        }

        let layer_1_rn = self.scratch.layer1_rn.forward(&out[0])?;
        let layer_2_rn = self.scratch.layer2_rn.forward(&out[1])?;
        let layer_3_rn = self.scratch.layer3_rn.forward(&out[2])?;
        let layer_4_rn = self.scratch.layer4_rn.forward(&out[3])?;

        let path4 = self.scratch.refine_net4.forward(&layer_4_rn)?;

        let res3_out = self.scratch.refine_net3.res_conv_unit1.forward(&layer_3_rn)?;
        let res3_out = path4.add(&res3_out)?;
        let path3 = self.scratch.refine_net3.forward(&res3_out)?;

        let res2_out = self.scratch.refine_net2.res_conv_unit1.forward(&layer_2_rn)?;
        let res2_out = path3.add(&res2_out)?;
        let path2 = self.scratch.refine_net2.forward(&res2_out)?;

        let res1_out = self.scratch.refine_net1.res_conv_unit1.forward(&layer_1_rn)?;
        let res1_out = path2.add(&res1_out)?;
        let path1 = self.scratch.refine_net1.forward(&res1_out)?;

        let out = self.scratch.output_conv1.forward(&path1)?;
        let out = out.interpolate2d(self.input_h, self.input_w)?;
        self.scratch.output_conv2.forward(&out)
    }
}

pub struct DepthAnythingV2 {
    pretrained: Arc<DinoVisionTransformer>,
    depth_head: DPTHead,
    conf: DepthAnythingV2Config,
}

impl DepthAnythingV2 {
    pub fn new(
        pretrained: Arc<DinoVisionTransformer>,
        conf: DepthAnythingV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depth_head = DPTHead::new(&conf, vb.pp("depth_head"))?;
        Ok(Self {
            pretrained,
            depth_head,
            conf,
        })
    }
}

impl Module for DepthAnythingV2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let features = self
            .pretrained
            .get_intermediate_layers(xs, &self.conf.layer_ids_vits)?;
        let depth = self.depth_head.forward(&features)?;
        depth * self.conf.max_depth
    }
}
