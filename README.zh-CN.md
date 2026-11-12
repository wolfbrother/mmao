# MMAO 优化算法 Python 包

MMAO 通过个体能量、公共资源池和反馈调节组织多智能体搜索。软件提供连续优化、旅行商、多维背包、动态优化以及分类特征选择与参数搜索接口。

相关论文《MMAO: A Unified Resource-Regulated Controller for Cross-Domain Optimization》已发表于 IEEE Access。论文作者为 Jinliang Xu、Bingqi Li、Liping Ma、Runkai Zhu、Fanjie Nie 和 Jin Li。[论文页面](https://ieeexplore.ieee.org/document/11783402) · [DOI](https://doi.org/10.1109/ACCESS.2026.3752083)

## 安装与首次运行

需要 Python 3.10 或更高版本。在源码目录安装当前维护版本。

```bash
python -m pip install -e ".[dev,io,plot]"
mmao --version
mmao continuous --problem sphere --dimension 5 --iterations 20 --seed 7 --summary-only
mmao tsp --benchmark berlin52 --iterations 20 --summary-only
mmao knapsack --benchmark mknap2 --iterations 10 --summary-only
mmao dynamic --iterations 20 --summary-only
mmao classification --dataset iris --classifier knn --iterations 4 --summary-only
```

PyPI 安装命令为 `python -m pip install mmao-opt`，取得的版本以 PyPI 实际发布情况为准。内置 Berlin52 可以直接使用。读取外部 TSPLIB 文件需要安装 `mmao-opt[io]`。

## 接入自己的目标函数

```python
from mmao import ContinuousProblem, MMAOContinuousConfig, optimize_continuous

def objective(x):
    return (x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2

problem = ContinuousProblem(
    name="my-objective", dimension=2,
    bounds=[(-5, 5), (-5, 5)], objective=objective,
)
result = optimize_continuous(
    problem, MMAOContinuousConfig(iterations=100, seed=7, max_evaluations=500)
)
print(result["best_position"], result["best_fitness"], result["nfev"])
```

连续优化接口执行最小化。自定义目标的已知最优值默认为未知，只有显式提供 `optimum_value` 时才计算最优差距。`max_evaluations` 计入初始化、感知、移动和新增个体的目标函数调用。一次移动使用三次评价，剩余预算不足时可能提前结束，实际计数不会超过上限。

## 分类数据与模型保存

通过 `ClassificationProblem(name="my-data", x=X, y=y)` 接入数值特征矩阵和分类标签。软件按种子划分训练、验证和测试集，搜索时使用训练与验证数据，最终使用选定的特征集合在训练和验证数据上重新训练。

```python
from sklearn.datasets import load_iris
from mmao import ClassificationProblem, MMAOClassificationConfig
from mmao import optimize_classification, save_classification_model

X, y = load_iris(return_X_y=True)
result = optimize_classification(
    ClassificationProblem(name="my-iris", x=X, y=y, classifier_name="knn"),
    MMAOClassificationConfig(iterations=4, seed=3, return_model=True),
)
print(result["selected_feature_indices"], result["test_metric"])
print(result["model"].predict(X[:3]))
save_classification_model(result, "outputs/iris.joblib")
```

保存的模型包含特征选择和分类流水线，预测时输入完整特征矩阵。可以用 `joblib.load` 读取自己保存的模型。模型文件依赖对应的 Python 与 scikit-learn 环境。JSON 导出保存数值、参数及运行记录，模型对象通过上述方法单独保存。

## 保存、批量运行与绘图

```bash
mmao continuous --dimension 5 --max-evaluations 500 --output outputs/run.json
python examples/benchmark_continuous.py --seeds 1 2 3 4 5 --max-evaluations 500 --output outputs/batch.json
python examples/custom_tasks.py
```

运行记录包含版本、配置、种子、耗时和问题指纹。批量示例输出每次运行结果、均值、标准差和基于 Student t 分布的均值置信区间，该区间采用样本独立、分布近似正态的假设。目标函数实现需要由使用者另外保留版本，问题指纹记录问题描述及输入数据。

```python
from mmao.plotting import plot_history
figure = plot_history(result, output_path="outputs/history.png")
```

绘图需要安装 `mmao-opt[plot]`。可展示目标值、资源池、种群数量和角色变化。连续优化的评价预算接口可用于固定预算实验，其他任务当前采用各自的迭代设置。

## 引用与复现

引用信息见 [CITATION.cff](CITATION.cff) 和 [references.bib](references.bib)。软件模块与研究内容的对应关系见 [论文覆盖说明](docs/paper-coverage.md)，实验运行及验证方法见 [复现说明](docs/reproducibility.md)。本次修复会改变部分任务的运行结果，比较实验应记录并统一所用软件版本。

## 版权与许可

版权所有 (c) 2026 Jinliang Xu。

作者为 Jinliang Xu，联系邮箱为 `jlxufly@gmail.com` 和 `xujinliang@caict.ac.cn`。

本项目采用 Apache License 2.0，完整许可条款与版权声明见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
