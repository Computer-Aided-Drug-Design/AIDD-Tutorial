LitMatter DeepChem
==================================================================

`Jupyter Notebook <https://github.com/ncfrey/litmatter/blob/main/LitDeepChem.ipynb>`_ 需要下载整个 `Github代码 <https://github.com/ncfrey/litmatter>`_ 然后再从 litmatter 文件夹中打开 LitDeepChem.ipynb

.. code-block:: Python

    git clone https://github.com/ncfrey/litmatter.git

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/litmatter/LitDeepChem.ipynb>`_

`Jupyter Notebook 中文翻译版下载 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/litmatter/LitDeepChem.ipynb>`_ 需要下载整个 `Github 代码 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files>`_ 然后再从 litmatter 文件夹中打开 LitDeepChem.ipynb

.. code-block:: Python

    git clone https://github.com/abdusemiabduweli/AIDD-Tutorial-Files.git

* 这本笔记本展示了如何使用 LitMatter 模板在 `MoleculeNet <https://arxiv.org/abs/1703.00564>`_ 数据集上加速 `DeepChem <https://github.com/deepchem/deepchem>`_ 模型训练。
* 在本例中，我们在 Tox21 数据集上训练一个简单的 DeepChem `TorchModel` 。
* 这里展示的训练工作流可以通过更改一个关键参数扩展到数百个 GPU！

.. code-block:: Python

    %load_ext autoreload
    %autoreload 2

    import torch

    import deepchem as dc

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                                seed_everything)

加载一个 `LitMolNet` 数据集
-------------------------------

任何来自 `deepchem.molnet` 的 MolNet 数据集可与 LitMatter 配合使用。具体的 MolNet 数据集和任何预处理步骤都可以在 `data.LitMolNet` 中定义。

.. code-block:: Python

    from lit_data.molnet_data import LitMolNet

    dm = LitMolNet(loader=dc.molnet.load_tox21, batch_size=16)
    dm.prepare_data()
    dm.setup()

实例化一个 `LitDeepChem` 模型
------------------------------

任何 `deepchem.models.torch_models.TorchModel` 可以与 LitMatter 一起使用。在这里，我们将在 PyTorch 中编写我们自己的自定义基本模型，并创建一个 `TorchModel` 。

.. code-block:: Python

    from lit_models.deepchem_models import LitDeepChem

    base_model = torch.nn.Sequential(
    torch.nn.Linear(1024, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 12),
    )

    torch_model = dc.models.TorchModel(base_model, loss=torch.nn.MSELoss())

    model = LitDeepChem(torch_model, lr=1e-2)

训练模型
--------------

在用多个 GPU 和多节点训练时，只需根据需要更改 `Trainer` 标志。

.. code-block:: Python

    trainer = Trainer(gpus=-1,  # use all available GPUs on each node
    #                   num_nodes=1,  # change to number of available nodes
    #                  accelerator='ddp',
                    max_epochs=5,
                    )

.. code-block:: Python

    trainer.fit(model, datamodule=dm)

就这样！通过改变 `num_nodes` 参数，训练可以分布在所有可用的 GPU 上。有关 HPC 集群上较长的训练作业，请参阅提供的示例批处理脚本。