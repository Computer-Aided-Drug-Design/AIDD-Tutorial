与PytorchLightning结合
==================================================================

`Jupyter Notebook <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/PytorchLightning_Integration.ipynb>`_

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/与PytorchLightning结合.ipynb>`_

`Jupyter Notebook 中文翻译版下载 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/与PytorchLightning结合.ipynb>`_

在本教程中，我们将介绍如何在 `pytorch-lightning <https://www.pytorchlightning.ai/>`_ 框架中设置 deepchem 模型。Lightning 是一个 pytorch 框架，它简化了使用 pytorch 模型的实验过程。pytorch lightning 提供的以下几个关键功能是 deepchem 用户可以发现有用的:

1. 多 gpu 训练功能：pytorch-lightning 提供简单的多 gpu、多节点训练。它还简化了跨不同集群基础设施（如 AWS、基于 slurm 的集群）启动多gpu、多节点作业的过程。

2. 减少 pytorch 的样板代码：lightning 负责处理诸如 `optimizer.zero_grad(), model.train(), model.eval()` 之类的细节。Lightning 还提供了实验日志功能，例如，无论在 CPU、GPU、多节点上进行训练，用户都可以在训练器内部使用 `self.log` 方法，它将适当地记录指标。

3. 可以加速训练的工具：半精确训练、梯度检查点、代码分析。

准备
-----

- 本笔记本假设你已经安装了deepchem，如果你没有，请执行 deepchem 安装页面的说明：https://deepchem.readthedocs.io/en/latest/get_started/installation.html。
- 安装 pytorchlightning 请参考 lightning 的主页：https://www.pytorchlightning.ai/

导入相关的包。

.. code-block:: Python

    import deepchem as dc
    from deepchem.models import GCNModel
    import pytorch_lightning as pl
    import torch
    from torch.nn import functional as F
    from torch import nn
    import pytorch_lightning as pl
    from pytorch_lightning.core.lightning import LightningModule
    from torch.optim import Adam
    import numpy as np
    import torch

Deepchem 例子
------------------------------

下面我们展示图卷积网络（GCN）的一个例子。请注意，这是一个简单的示例，它使用 GCNModel 从输入序列预测标签。在这个例子中，我们没有展示 deepchem 的完整功能，因为我们想要重组 deepchem 代码，并对其进行调整，以便能够轻松插入 pytorch-lightning。这个例子的灵感来自 `GCNModel` `文档 <https://github.com/deepchem/deepchem/blob/a68f8c072b80a1bce5671250aef60f9cc8519bec/deepchem/models/torch_models/gcn.py#L200>`_ 。

**准备数据集**：为了训练我们的deepchem模型，我们需要一个可以用来训练模型的数据集。下面我们为本教程准备了一个示例数据集。下面我们也直接使用特征器编码数据集编码。

.. code-block:: Python

    smiles = ["C1CCC1", "CCC"]
    labels = [0., 1.]
    featurizer = dc.feat.MolGraphConvFeaturizer()
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X, y=labels)

**设置模型**：现在我们初始化我们将在训练中使用的图卷积网络模型。

.. code-block:: Python

    model = GCNModel(
        mode='classification',
        n_tasks=1,
        batch_size=2,
        learning_rate=0.001
    )

**训练模型**：在我们的训练数据集上拟合模型，还指定要运行的epoch的数量。

.. code-block:: Python

    loss = model.fit(dataset, nb_epoch=5)
    print(loss)

Pytorch-Lightning + Deepchem 示例
------------------------------------

现在我们来看一个 GCN 模型适用于 Pytorch-Lightning 的例子。使用 Pytorch-Lightning 有两个重要的组成部分：

1. `LightningDataModule` ：该模块定义如何准备数据并将其输入到模型中，以便模型可以使用它进行训练。该模块定义了训练数据加载器函数，训练器直接使用该函数为 `LightningModule` 生成数据。要了解有关 `LightningDataModule` 的更多信息，请参阅 `datamodules 文档 <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_ 。

2. `LightningModule` ：这个模块为我们的模型定义了训练和验证步骤。我们可以使用这个模块根据超参数初始化我们的模型。我们可以直接使用许多样板函数来跟踪我们的实验，例如，我们可以使用 `self.save_hyperparameters()` 方法来保存我们用于训练的所有超参数。有关如何使用该模块的详细信息，请参阅 `lightningmodules 文档 <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ 。

**设置 torch 数据集**：请注意，这里我们需要创建一个自定义的 `SmilesDataset`，以便我们可以轻松地与 deepchem 特征器交互。为这个交互,我们需要定义一个整理方法,这样我们就可以创建批次数据集。

.. code-block:: Python

    # prepare LightningDataModule
    class SmilesDataset(torch.utils.data.Dataset):
        def __init__(self, smiles, labels):
            assert len(smiles) == len(labels)
            featurizer = dc.feat.MolGraphConvFeaturizer()
            X = featurizer.featurize(smiles)
            self._samples = dc.data.NumpyDataset(X=X, y=labels)
            
        def __len__(self):
            return len(self._samples)
            
        def __getitem__(self, index):
            return (
                self._samples.X[index],
                self._samples.y[index],
                self._samples.w[index],
            )
        
        
    class SmilesDatasetBatch:
        def __init__(self, batch):
            X = [np.array([b[0] for b in batch])]
            y = [np.array([b[1] for b in batch])]
            w = [np.array([b[2] for b in batch])]
            self.batch_list = [X, y, w]
            
            
    def collate_smiles_dataset_wrapper(batch):
        return SmilesDatasetBatch(batch)

**创建 lightning 模块**：在本部分中，我们创建 GCN 特定的 lightning 模块。该类指定训练步骤的逻辑流。我们还为训练流创建所需的模型、优化器和损失。

.. code-block:: Python

    # prepare the LightningModule
    class GCNModule(pl.LightningModule):
        def __init__(self, mode, n_tasks, learning_rate):
            super().__init__()
            self.save_hyperparameters(
                "mode",
                "n_tasks",
                "learning_rate",
            )
            self.gcn_model = GCNModel(
                mode=self.hparams.mode,
                n_tasks=self.hparams.n_tasks,
                learning_rate=self.hparams.learning_rate,
            )
            self.pt_model = self.gcn_model.model
            self.loss = self.gcn_model._loss_fn
            
        def configure_optimizers(self):
            return self.gcn_model.optimizer._create_pytorch_optimizer(
                self.pt_model.parameters(),
            )
        
        def training_step(self, batch, batch_idx):
            batch = batch.batch_list
            inputs, labels, weights = self.gcn_model._prepare_batch(batch)
            outputs = self.pt_model(inputs)
            
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
        
            if self.gcn_model._loss_outputs is not None:
                outputs = [outputs[i] for i in self.gcn_model._loss_outputs]
        
            loss_outputs = self.loss(outputs, labels, weights)
            
            self.log(
                "train_loss",
                loss_outputs,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="mean",
                prog_bar=True,
            )
            
            return loss_outputs

**创建相关对象**

.. code-block:: Python

    # create module objects
    smiles_datasetmodule = SmilesDatasetModule(
        train_smiles=["C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC"],
        train_labels=[0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
        batch_size=2,
    )

    gcnmodule = GCNModule(
        mode="classification",
        n_tasks=1,
        learning_rate=1e-3,
    )

Lightning 训练器
--------------------------

Trainer 是构建在 `LightningDataModule` 和 `LightningModule` 之上的包装器。当构建 lightning 训练器时，你还可以指定 epoch 的数量，运行的最大步数，gpu 的数量，用于训练器的节点数量。Lightning trainer 充当分布式训练设置的包装器，这样你就能够简单地构建模型以本地运行。

.. code-block:: Python

    trainer = pl.Trainer(
        max_epochs=5,
    )

**调用 fit 函数运行模型训练**

.. code-block:: Python

    # train
    trainer.fit(
        model=gcnmodule,
        datamodule=smiles_datasetmodule,
    )

