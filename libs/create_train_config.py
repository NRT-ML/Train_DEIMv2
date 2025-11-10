import yaml
import json
import os, sys
from pathlib import Path
import copy

# YAMLのアンカー参照（*id001など）を無効化するカスタムDumper
class NoAliasDumper(yaml.SafeDumper):
    """アンカーとエイリアスを使わずにYAMLをダンプするカスタムDumper
    
    通常、yaml.dump()は同じオブジェクトが複数回参照される場合、
    アンカー（&id001）とエイリアス（*id001）を使って重複を避けますが、
    このDumperはそれを無効化し、値を毎回展開して出力します。
    """
    def ignore_aliases(self, data):
        """すべてのデータに対してエイリアスを無視する"""
        return True

BASE_DIR = Path(__file__).absolute().parent.parent
DEIMV2_CFG_DIR = BASE_DIR / "DEIMv2/configs"

# 各モデルの推奨入力サイズ (height, width)
MODEL_INPUT_SIZES = {
    # DEIMv2 - HGNetv2シリーズ
    'deimv2_hgnetv2_x_coco': [640, 640],
    'deimv2_hgnetv2_l_coco': [640, 640],
    'deimv2_hgnetv2_m_coco': [640, 640],
    'deimv2_hgnetv2_s_coco': [640, 640],
    'deimv2_hgnetv2_n_coco': [640, 640],
    'deimv2_hgnetv2_pico_coco': [640, 640],
    'deimv2_hgnetv2_femto_coco': [416, 416],
    'deimv2_hgnetv2_atto_coco': [320, 320],
    
    # DEIMv2 - DINOv3シリーズ
    'deimv2_dinov3_x_coco': [640, 640],
    'deimv2_dinov3_l_coco': [640, 640],
    'deimv2_dinov3_m_coco': [640, 640],
    'deimv2_dinov3_s_coco': [640, 640],
    
    # DEIM - HGNetv2シリーズ
    'deim_hgnetv2_x_coco': [640, 640],
    'deim_hgnetv2_l_coco': [640, 640],
    'deim_hgnetv2_m_coco': [640, 640],
    'deim_hgnetv2_s_coco': [640, 640],
    'deim_hgnetv2_n_coco': [640, 640],
    
    # D-FINE - HGNetv2シリーズ
    'dfine_hgnetv2_x_coco': [640, 640],
    'dfine_hgnetv2_l_coco': [640, 640],
    'dfine_hgnetv2_m_coco': [640, 640],
    'dfine_hgnetv2_s_coco': [640, 640],
    'dfine_hgnetv2_n_coco': [640, 640],
    
    # DEIM - RT-DETRv2シリーズ
    'deim_r101vd_60e_coco': [640, 640],
    'deim_r50vd_60e_coco': [640, 640],
    'deim_r50vd_m_60e_coco': [640, 640],
    'deim_r34vd_120e_coco': [640, 640],
    'deim_r18vd_120e_coco': [640, 640],
    
    # RT-DETRv2シリーズ
    'rtdetrv2_r101vd_6x_coco': [640, 640],
    'rtdetrv2_r50vd_6x_coco': [640, 640],
    'rtdetrv2_r50vd_m_7x_coco': [640, 640],
    'rtdetrv2_r34vd_120e_coco': [640, 640],
    'rtdetrv2_r18vd_120e_coco': [640, 640],
}

class CreateTrainConfig:
    def __init__(self, cfg_path: str):
        print(DEIMV2_CFG_DIR)
        self.cfg_path = cfg_path
        self.train_cfg_path: str = Path(cfg_path).parent / 'train_config.yaml'
        
        self.cfg: dict | None = None
        self.train_cfg:dict | None = None
        self._generate_train_cfg(self.train_cfg_path)

    def _generate_train_cfg(self, output_path):
        self._load_cfg()
        self._copy_to_train_cfg()
        self._add_include()
        self._add_num_classes()
        self._add_epochs()
        self._add_image_size()
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.train_cfg, 
                f, 
                Dumper=NoAliasDumper,  # カスタムDumperを使用してアンカー参照を無効化
                default_flow_style=False,  # ブロックスタイルで出力
                allow_unicode=True,  # 日本語などのUnicode文字を許可
                sort_keys=False  # キーをソートしない（元の順序を保持）
            )

    def _load_cfg(self):
        """ユーザ設定config.yamlを読み込む"""
        with open(self.cfg_path, 'rb') as file:
            self.cfg = yaml.safe_load(file)

    def _copy_to_train_cfg(self):
        """self.cfgから必要な情報のみをself.train_cfgにコピー"""
        self.train_cfg = copy.deepcopy(self.cfg)
        if 'model' in self.train_cfg:
            del self.train_cfg['model']
        if "epochs" in self.train_cfg:
            del self.train_cfg['epochs']
    
    def _add_include(self):
        """読み込むモデルの設定ファイルを__include__に追加"""
        model_name = self.cfg['model']
        model_cfg_path = list(DEIMV2_CFG_DIR.glob(f'**/{model_name}.yml'))[0]
        self.train_cfg['__include__'] = [str(model_cfg_path)]
    
    def _add_num_classes(self):
        """COCOフォーマットのアノテーションファイルからクラス数を取得し、train_cfgに追加"""
        ann_json_path = self.cfg['train_dataloader']['dataset']['ann_file']
        num_classes = self._get_num_classes(BASE_DIR / ann_json_path)
        print(f"num_classes: {num_classes}")
        self.train_cfg['num_classes'] = num_classes + 1  # 背景クラスを含む

    def _get_num_classes(self, json_path: str) -> int:
        """COCOフォーマットのアノテーションファイルからクラス数を取得"""
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        return len(coco_data['categories'])
    
    def _add_epochs(self):
        """エポック数をtrain_cfgに追加"""
        epochs = self.cfg.get('epochs')
        if epochs:
            self.train_cfg['epoches'] = epochs # 総エポック数

            # flat epoch
            flat_epoch = max(1, int(epochs * 0.5))
            self.train_cfg['flat_epoch'] = flat_epoch

            # no aug epoch
            no_aug_epoch = int(epochs * 0.1)
            self.train_cfg['no_aug_epoch'] = no_aug_epoch
            
            # mixup epochs
            start_epoch = max(1, int(epochs * 0.04))  # At least epoch 1
            mixup_epochs = [start_epoch, flat_epoch]
            
            # train_dataloaderの設定を初期化
            self.train_cfg.setdefault("train_dataloader", {}).setdefault("collate_fn", {})
            self.train_cfg["train_dataloader"].setdefault("dataset", {}).setdefault("transforms", {}).setdefault("policy", {})
            # mixup_epochs
            self.train_cfg["train_dataloader"]["collate_fn"]["mixup_epochs"] = mixup_epochs
            # stop epoch
            self.train_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = epochs - no_aug_epoch
            # copy blend epoch
            self.train_cfg["train_dataloader"]["collate_fn"]["copyblend_epochs"] = [start_epoch, epochs - no_aug_epoch]
                
            # data aug epochs
            self.train_cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] = \
                [start_epoch, flat_epoch, epochs - no_aug_epoch]
            
            # DEIMCriterionの設定を初期化
            self.train_cfg.setdefault("DEIMCriterion", {}).setdefault("matcher", {})
            # matcher_change_epoch
            self.train_cfg["DEIMCriterion"]["matcher"]["matcher_change_epoch"] = int((epochs - no_aug_epoch)*0.9)

        
        # # warmup iter
        # n_imgs = self._count_images(self.train_cfg["train_dataloader"]["dataset"]["img_folder"])
        # batch_size = self.train_cfg["train_dataloader"]["total_batch_size"]
        # iters_per_epoch = n_imgs // batch_size

        # warmup_iter = int(iters_per_epoch * epochs * 0.05)
        # min_warmup_iter = int(iters_per_epoch)
        # warmup_iter = max(warmup_iter, min_warmup_iter)

        # # ema warmup iter
        # ema_warmups = int(iters_per_epoch * epochs * 0.05)
        # min_ema_warmups = int(iters_per_epoch)
        # ema_warmups = max(ema_warmups, min_ema_warmups)


    # def _count_images(folder: str) -> int:
    #     """画像数をカウントします。"""
    #     n_imgs = len([f for f in Path(folder).iterdir() if f.is_file()])
    #     return n_imgs
    
    def _add_image_size(self):
        """モデルに適した入力サイズをtrain_cfgに追加
        
        モデル名から適切な入力サイズを取得し、以下の設定に反映します:
        - eval_spatial_size: 評価時の入力サイズ
        - train_dataloader.dataset.transforms.ops内のResizeサイズ
        - val_dataloader.dataset.transforms.ops内のResizeサイズ (存在する場合)
        - train_dataloader.collate_fn.base_size (存在する場合)
        """
        model_name = self.cfg['model']

        auto_change_size = False
        
        # モデル名から入力サイズを取得
        input_size = MODEL_INPUT_SIZES.get(model_name, [640, 640])

        # transforms
        for data_loader in ['train_dataloader', 'val_dataloader']:
            transforms = self.train_cfg[data_loader].get('dataset', {}).get('transforms', {}).get('ops', [])
            for op in transforms:
                if op['type'] == 'Resize':
                    op['size'] = input_size
                    auto_change_size = True
                elif op['type'] == "Mosaic":
                    op['output_size'] = input_size[0]//2
                    auto_change_size = True

        # eval_spatial_size
        if auto_change_size:
            self.train_cfg['eval_spatial_size'] = input_size