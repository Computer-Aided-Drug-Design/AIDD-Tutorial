使用DeepChem训练第一个模型
================================

`Jupyter Notebook <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb>`_

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/%E4%BD%BF%E7%94%A8DeepChem%E8%AE%AD%E7%BB%83%E7%AC%AC%E4%B8%80%E4%B8%AA%E6%A8%A1%E5%9E%8B.ipynb>`_

`Jupyter Notebook 中文翻译版下载 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/%E4%BD%BF%E7%94%A8DeepChem%E8%AE%AD%E7%BB%83%E7%AC%AC%E4%B8%80%E4%B8%AA%E6%A8%A1%E5%9E%8B.ipynb>`_

深度学习可以用来解决许多类型的问题，但基本工作流程通常是相同的。以下是你可以遵循的典型步骤。

1. 选择你用来训练模型的数据集(训练集，如果没有合适的现有数据集，则建立一个新的数据集)。
2. 建立模型。
3. 使用训练集训练模型。
4. 使用独立于训练集的验证集评估模型。
5. 使用该模型对新数据进行预测。

有了DeepChem，每一个步骤都可以少到只有一两行Python代码。在本教程中，我们将通过一个基本示例演示解决现实科学问题的完整工作流。

我们要解决的问题是根据小分子的化学公式预测其溶解度。这是药物开发中一个非常重要的特性：如果一种拟用药物的溶解性不够，很可能进入患者血液的药物的量不够，从而产生不了治疗效果。我们需要的第一个东西是真实分子溶解度的数据集。DeepChem的核心组件之一，MoleculeNet，是多样化学分子数据集的集合。对于本教程，我们可以使用Delaney溶解度数据集。此数据集中的溶解度数据是以log（溶解度）表示，其中溶解度以摩尔/升为单位进行测量。

.. code-block:: Python

    import deepchem as dc
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

我现在不会对这段代码解释太多。我们将在后面的教程中看到许多类似的例子。有两个细节我想让你注意一下。首先，注意传递给' load_delaney() '函数的' featurizer '参数。分子可以用多种方式表示。因此，我们告诉它我们想要使用哪种方式表示，或者用更专业的语言来说，如何“特征器（featurizer）”数据。其次，注意我们实际上得到了三个不同的数据集:训练集、验证集和测试集（分别对应 train_dataset, valid_dataset, test_dataset ）。在标准的深度学习工作流程中，每一个数据集都有不同的功能。

现在我们有了数据集，下一步是建立一个模型。我们将使用一种特殊的模型，称为“图卷积网络（graph convolutional network）”，简称为“graphconv”。

.. code-block:: Python

    model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)

在这里我也不会过多地介绍上述代码。后面的教程将会提供更多关于“GraphConvModel”，以及DeepChem提供的其他类型的模型的信息。

我们现在需要使用训练集训练模型。我们只需给它一个数据集，然后告诉它要进行多少次训练(也就是说，要多少遍完整的使用训练集)。

.. code-block:: Python

    model.fit(train_dataset, nb_epoch=100)

如果一切进展顺利，我们现在应该有一个完全训练好的模型了!但是这个模型靠谱吗?为了找出答案，我们必须使用验证集评估模型。我们通过选择一个评估指标并在模型上调用“evaluate()”来做到这一点。对于这个例子，让我们使用皮尔逊相关（the Pearson correlation），也称为 :math:`r^2` ，作为我们的评估指标。我们可以在训练集和测试集上对它进行评估。

.. code-block:: Python

    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print("Training set score:", model.evaluate(train_dataset, [metric], transformers))
    print("Test set score:", model.evaluate(test_dataset, [metric], transformers))

注意，它在训练集上的得分高于测试集。对比于测试集，模型通常在训练集表现得更好。这被称为“过拟合”，这也是为什么使用独立的测试集评估模型是至关重要的原因。

我们的模型在测试集上仍然有相当不错的表现。作为比较，一个产生完全随机输出的模型的相关性为0，而一个做出完美预测的模型的相关性为1。我们的模型做得很好，所以现在我们可以用它来预测其他我们关心的分子。

因为这只是一个教程，我们没有任何其他的分子我们特别想要预测，让我们只使用测试集中的前十个分子进行预测。对于每一个分子，我们打印出其化学结构(使用SMILES字符串表示)，真实的log(溶解度)值，和预测的log(溶解度)值。

.. code-block:: Python

    predictions = model.predict_on_batch(test_dataset.X[:10])
    for molecule, test_solubility, prediction in zip(test_dataset.ids, test_dataset.y, predictions):
        print(molecule, test_solubility, prediction)

完。