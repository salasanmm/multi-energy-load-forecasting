class BaseConfig:
    def __init__(self):
        self.task_name = "long_term_forecast"
        self.is_training = 1
        self.model_id = "test"
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 72
        self.embed = "timeF"
        self.activation = "gelu"
        self.dropout = 0.1
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = "dft_decomp"
        self.moving_avg = 24
        self.factor = 1
        self.lr = 1.0e-3
        self.batchsize = 256
        self.epochs = 60
        self.optimizer_weight_decay = 1.0e-5
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 4


class TimeMixerConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pred_in_len = 168
        self.individual = True
        self.down_sampling_window = 2
        self.channel_independence = False
        self.e_layers = 3
        self.down_sampling_layers = 2
        self.enc_in = 12
        self.embed = 64
        self.d_model = 64
        self.freq = "h"
        self.c_out = 12
        self.num_class = 10
        self.top_k = 5
        self.moving_avg = 3
        self.down_sampling_method = "avg"
        self.d_ff = 64
        self.use_pgf = True
        self.pgf_hidden = 32
        self.pgf_dropout = 0.1
        self.pgf_gate_bias = -3.0
        self.use_crd = True
        self.crd_hidden = 24
        self.crd_dropout = 0.1
        self.crd_gate_bias = -3.0
        self.use_ph = True
        self.ph_hidden = 3
        self.ph_kernel = 3
        self.ph_dropout = 0.1
        self.ph_gate_bias = -3.0
        self.use_anchor = True
        self.anchor_hidden = 32
        self.anchor_dropout = 0.1
        self.anchor_gate_bias = -3.0
        self.anchor_init_scale = 0.02
        self.use_weekly_anchor = False
        self.weekly_anchor_hidden = 32
        self.weekly_anchor_dropout = 0.1
        self.weekly_anchor_gate_bias = -3.0
        self.weekly_anchor_init_scale = 0.02
        self.use_haf = True
        self.haf_init_scale = 0.02
        self.use_mssm = True
        self.use_mstm = True
        self.use_out_residual = True
        self.use_regression = True
        self.channel_loss_weights = [1.0, 1.0, 1.0]
        self.exclude_columns = []
        self.train_mae_weight = 0.0005
        self.train_ratio = 0.8
        self.valid_ratio = 0.1
        self.test_ratio = 0.1
        self.experiment_tag = "dft_top5_811"
        self.seasonal_eval = False
        self.warm_start_path = None
        self.protect_best_path = None
        self.protect_best_mae = float("inf")
        self.train_select_metric = "valid_mae"
        self.output_best_test_path = "./output/TimeMixer_24_168/model/model_lnn_best_test.pt"
        self.use_overlap_smooth = False
        self.overlap_smooth_blend = 0.0
        self.eval_only = False
        self.weightdecay = 0.5
        self.decaypatience = 4


def get_config(model_name: str):
    if model_name != "TimeMixer":
        raise ValueError(f"Unsupported model name: {model_name}")
    return TimeMixerConfig()
