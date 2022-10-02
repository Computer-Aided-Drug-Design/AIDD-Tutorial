对 MoleculeNet 的介绍
================================

`Jupyter Notebook <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/An_Introduction_To_MoleculeNet.ipynb>`_

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/%E5%AF%B9%20MoleculeNet%20%E7%9A%84%E4%BB%8B%E7%BB%8D.ipynb>`_

`Jupyter Notebook 中文翻译版 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/%E5%AF%B9%20MoleculeNet%20%E7%9A%84%E4%BB%8B%E7%BB%8D.ipynb>`_

DeepChem最强大的功能之一是它自带“电池”，也就是说，它自带数据集。DeepChem开发者社区维护MoleculeNet[1]数据集套件，它包含了大量不同的科学数据集，用于机器学习应用。最初的MoleculeNet套件有17个主要关注分子性质的数据集。在过去的几年里，MoleculeNet已经发展成为一个更广泛的科学数据集集合，以促进科学机器学习工具的广泛使用和发展。

这些数据集与DeepChem套件的其他部分集成在一起，因此你可以通过 **dc.molnet** 子模块中的函数方便地访问这些数据集。在学习本系列教程的过程中，你已经看到了这些加载器（loaders）的一些示例。MoleculeNet套件的完整文档可在我们的文档[2]中找到。

[1] Wu, Zhenqin, et al. "MoleculeNet: a benchmark for molecular machine learning." Chemical science 9.2 (2018): 513-530.

[2] https://deepchem.readthedocs.io/en/latest/moleculenet.html

MoleculeNet的概述
----------------------------

在上两个教程中，我们加载了分子溶解度——Delaney数据集。让我们再加载一次。

.. code-block:: Python

    import deepchem as dc

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

注意我们调用的加载器函数 **dc.molnet.load_delaney** 在 **dc.molnet** 的子模块。让我们看一看可供我们使用的加载器函数的完整名单。

.. code-block:: Python

    [method for method in dir(dc.molnet) if "load_" in method ]

MoleculeNet加载器由DeepChem社区积极维护，我们致力于向集合中添加新的数据集。让我们看看今天在MoleculeNet中有多少数据集。

.. code-block:: Python

    len([method for method in dir(dc.molnet) if "load_" in method ])

MoleculeNet数据集类别
--------------------------------

MoleculeNet 中有很多不同的数据集。让我们快速概述一下可用的不同类型的数据集。我们将把数据集分成不同的类别，并列出属于这些类别的加载器。更多关于这些数据集的细节可以在 https://deepchem.readthedocs.io/en/latest/moleculenet.html 上找到。最初的MoleculeNet论文[1]提供了标记为 **V1** 的数据集的详细信息。所有剩下的数据集都是 **V2** ，在论文中没有记录。

量子力学数据集
:::::::::::::::::

MoleculeNet的量子力学数据集包含各种量子力学特性预测任务。目前的量子力学数据集包括QM7、QM7b、QM8、QM9。相关的加载器是：

- `dc.molnet.load_qm7 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_qm7>`_ : V1
- `dc.molnet.load_qm7b_from_mat <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_qm7>`_ : V1
- `dc.molnet.load_qm8 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_qm8>`_ : V1
- `dc.molnet.load_qm9 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_qm9>`_ : V1

物理化学数据集
:::::::::::::::::::::::::::::::

物理化学数据集包含预测分子的各种物理性质的各种任务。

- `dc.molnet.load_delaney <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_delaney>`_ ：V1 这个数据集在原论文中也称为ESOL。
- `dc.molnet.load_sampl <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_sampl>`_ ：V1 这个数据集在原论文中也称为FreeSolv。
- `dc.molnet.load_lipo <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_lipo>`_ ：V1 这个数据集在原文中也被称为Lipophilicity。
- `dc.molnet.load_thermosol <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_thermosol>`_ ：V2
- `dc.molnet.load_hppb <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_hppb>`_ ：V2
- `dc.molnet.load_hopv <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_hopv>`_ ：V2 该数据集来自最近出版的一篇文章 [3]

化学反应数据集
::::::::::::::::

这些数据集包含了用于计算逆向合成/正向合成的化学反应数据集。

- `dc.molnet.load_uspto <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_uspto>`_

生物化学/生物物理数据集
::::::::::::::::::::::::::::

这些数据集是从各种生化/生物物理数据集中提取的，这些数据集包括比如化合物与蛋白质的结合亲和力等数据。

- `dc.molnet.load_pcba <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_pcba>`_ ：V1
- `dc.molnet.load_nci <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_nci>`_ ：V2
- `dc.molnet.load_muv <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_muv>`_ ：V1
- `dc.molnet.load_hiv <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_hiv>`_ ：V1
- `dc.molnet.load_ppb <https://deepchem.readthedocs.io/en/latest/moleculenet.html#ppb-datasets>`_ ：V2。
- `dc.molnet.load_bace_classification <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bace_classification>`_ ：V1 这个加载器加载原MoleculeNet论文中的分类BACE数据集的程序。
- `dc.molnet.load_bace_regression <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bace_regression>`_ ：V1 这个加载器加载原MoleculeNet论文中的回归分析BACE数据集的程序。
- `dc.molnet.load_kaggle <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_kaggle>`_ ：V2 该数据集来自Merck的药物发现kaggle竞赛内容，在[4]中进行了描述。
- `dc.molnet.load_factors <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_factors>`_ ：V2 该数据集来自[4]。
- `dc.molnet.load_uv <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_uv>`_ ：V2 该数据集来自[4]。
- `dc.molnet.load_kinase <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_kinase>`_ ：V2 该数据集来自[4]。

分子目录数据集
::::::::::::::::::::::::::::::

这些数据集提供的分子数据集除了原始的SMILES公式或结构外没有相关的性质。这种类型的数据集对于建立生成模型任务非常有用。

- `dc.molnet.load_zinc15 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_zinc15>`_ ：V2
- `dc.molnet.load_chembl <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_chembl>`_ ：V2
- `dc.molnet.load_chembl25 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#chembl25-datasets>`_ ：V2

生理学数据集
::::::::::::::

这些数据集包含关于化合物如何与人类患者相互作用的生理特性的实验数据。

- `dc.molnet.load_bbbp <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bbbp>`_ ：V1
- `dc.molnet.load_tox21 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_tox21>`_ ：V1
- `dc.molnet.load_toxcast <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_toxcast>`_ ：V1
- `dc.molnet.load_sider <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_sider>`_ ：V1
- `dc.molnet.load_clintox <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_clintox>`_ ：V1
- `dc.molnet.load_clearance <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_clearance>`_ ：V2

结构生物学数据集
::::::::::::::::::::::

这些数据集包含大分子的三维结构和相关的性质。

- `dc.molnet.load_pdbbind <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_pdbbind>`_ ：V1


显微术数据集
::::::::::::::::

这些数据集包含显微术图像数据集，通常是细胞系。这些数据集并没有出现在最初的MoleculeNet论文中。

- `dc.molnet.load_bbbc001 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bbbc001>`_ ：V2
- `dc.molnet.load_bbbc002 <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bbbc002>`_ ：V2
- `dc.molnet.load_cell_counting <https://deepchem.readthedocs.io/en/latest/moleculenet.html#cell-counting-datasets>`_ ：V2

材料属性数据集
::::::::::::::::

这些数据集包含关于各种材料的性能的数据。

- `dc.molnet.load_bandgap <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_bandgap>`_ ：V2
- `dc.molnet.load_perovskite <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_perovskite>`_ ：V2
- `dc.molnet.load_mp_formation_energy <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_mp_formation_energy>`_ ：V2
- `dc.molnet.load_mp_metallicity <https://deepchem.readthedocs.io/en/latest/moleculenet.html#deepchem.molnet.load_mp_metallicity>`_ ：V2


[3] Lopez, Steven A., et al. "The Harvard organic photovoltaic dataset." Scientific data 3.1 (2016): 1-7.

[4] Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

MoleculeNet加载器（Loaders）的解释
----------------------------------------

所有的MoleculeNet加载器函数都采用 **dc.molnet.load_X** 的形式。加载器函数返回一个元组 **(任务，数据集，转换器)(tasks, datasets, transformers)** 。让我们遍历每个返回值并解释我们得到了什么:

1. `任务（tasks）`：这是一个任务名称列表。MoleculeNet中的许多数据集都是“多任务”的。也就是说，一个给定的数据点有多个与之相关的标签。这些对应于与这个数据点相关的不同测量值或值。
2. `数据集（datasets）`：这是一个元组包含三个 `dc.data.Dataset` 对象 `(训练，验证，测试)` 。这些对应于这个MoleculeNet数据集的训练、验证和测试集。
3. `转换器（transformers）`：这是一个在处理期间应用于此数据集的 `dc.trans.Transformer` 对象列表。

这有点抽象，所以让我们看看我们上面调用的 **dc.molnet.load_delaney** 函数的这些返回值。让我们从 **任务（tasks）** 开始。

.. code:: Python

    print(tasks)

我们在这个数据集中有一个任务，它对应于测量的 log(溶解度)，单位为mol/L。现在让我们来看看 **数据集（datasets）** ：

.. code:: Python

    print(datasets)

正如我们前面提到的，我们看到 **datassets** 是一个包含3个数据集的元组。我们把它们分开。

.. code:: Python

    train, valid, test = datasets
    print(train)
    print(valid)
    print(test)

让我们来看看 **train** 数据集中的一个数据点。

.. code:: Python

    print(train.X[0])

注意，这是一个由 **dc.feat.ConvMolFeaturizer** 生成的 **dc.feat.mol_graphs.ConvMol** 对象。稍后我们将更多地讨论如何选择特征化（featurization）。最后让我们来看看 **transformers** ：

.. code:: Python

    print(transformers)

我们看到一个转换器(transformer)被应用了， **dc.trans.NormalizationTransformer** 。

在阅读完这篇描述之后，你可能想知道在底层做了哪些选择。正如我们之前简要提到的，可以使用不同的“featurizer”来处理数据集。在这儿，我们能选择如何特征化吗?此外，如何将源数据集分割为训练/验证/测试三个不同的数据集?

你可以使用 **featurizer** 和 **splitter** 关键字参数并传入不同的字符串。“featurizer”通常可能的选择是“ECFP”，“GraphConv”，“Weave”和“smiles2img”，对应于 **dc.feat.CircularFingerprint** 、 **dc.feat.ConvMolFeaturizer** 、 **dc.feat.WeaveFeaturizer** 和 **dc.feat.SmilesToImage** 。splitter的常见可能选项是“None”，“index”，“random”，“scaffold”和“stratified”，对应于no split， **dc.splits.IndexSplitter** , **dc.splits.RandomSplitter** , **dc.splits.SingletaskStratifiedSplitter** 。我们还没有讨论分离器，但直观地说，它们是一种基于不同标准划分数据集的方法。我们将在以后的教程中详细介绍。

除了字符串，你还可以传入任何 **Featurizer** 或 **Splitter** 对象。这是非常有用的，例如，Featurizer的构造参数可以被用来定制它的行为。

.. code:: Python

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="ECFP", splitter="scaffold")
    (train, valid, test) = datasets
    print(train)
    print(train.X[0])

注意，与前面的调用不同，我们有了由 `dc.feat.CircularFingerprint` 生成的numpy数组。而不是 `dc.feat.ConvMolFeaturizer` 生成的 `ConvMol` 对象。

自己试试吧。尝试调用MoleculeNet来加载一些其他的数据集，并使用不同的 featurizer/splitter 选项进行实验，看看会发生什么!
