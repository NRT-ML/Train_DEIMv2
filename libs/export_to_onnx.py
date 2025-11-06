import os
import sys
import onnx
import onnxsim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from DEIMv2.engine.core import YAMLConfig


def export_to_onnx(config_path: str,
                   resume: str,
                   opset: int = 17,
                   check: bool = True,
                   simplify: bool = True
                   ):
    """
    onnx変換。

    Args:
        config_path (str): YAMLの設定ファイルのパス
        resume (str): 学習済みモデルの重み(.pthファイル)のパス
        opset (int, optional): ONNXのopsetバージョン。デフォルトは17
        check (bool, optional): エクスポート後にONNXモデルの検証を行うか。デフォルトはTrue
        simplify (bool, optional): ONNXモデルを簡略化するか。デフォルトはTrue
    """
    cfg = YAMLConfig(config_path, resume=resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(32, 3, *img_size)
    size = torch.tensor([img_size])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    output_file = resume.replace('.pth', '.onnx') if resume else 'model.onnx'

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )

    if check:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if simplify:
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')