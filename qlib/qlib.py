import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.eva.alpha import risk_analysis
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils import init_instance_by_config

# 初始化 Qlib，使用中国市场数据
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)

# 数据集处理器配置
handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
    "features": [
        # 基础特征
        {"feature": "Ref($close, -1) / $close - 1", "name": "Return_1"},
        {"feature": "Ref($close, -5) / $close - 1", "name": "Return_5"},
        # 技术指标特征
        {"feature": "Mean($close, 5) / $close - 1", "name": "MA_5"},
        {"feature": "Mean($close, 10) / $close - 1", "name": "MA_10"},
        {"feature": "Mean($close, 20) / $close - 1", "name": "MA_20"},
        {"feature": "Std($close, 5) / $close - 1", "name": "STD_5"},
    ],
}

# 数据集配置
dataset = DatasetH(
    handler=DataHandlerLP(**handler_config),
    segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
    },
)

# 模型配置
model = LGBModel(
    loss="mse",
    learning_rate=0.05,
    n_estimators=1000,
    num_leaves=64,
    feature_fraction=0.7,
    subsample=0.8,
    nthread=20,
)

# 创建实验
with R.start(experiment_name="lightgbm_exp"):
    # 训练模型
    R.log_params(**handler_config)
    model.fit(dataset)
    R.save_objects(trained_model=model)
    
    # 记录信号
    sr = SignalRecord(model, dataset, rec_key="signal")
    sr.generate()

    # 分析信号
    sa = SigAnaRecord(rec_key="signal")
    sa.generate()

    # 配置策略
    strategy_config = {
        "topk": 50,
        "n_drop": 5,
    }
    strategy = TopkDropoutStrategy(**strategy_config)

    # 记录交易结果
    par = PortAnaRecord(experiment_name="lightgbm_exp", strategy=strategy)
    par.generate()

# 风险分析
pred_score = R.get_recorder().list_recs()[-1]
analysis_df = risk_analysis(pred_score["predict"]["label"], pred_score["predict"]["score"])

print(analysis_df)