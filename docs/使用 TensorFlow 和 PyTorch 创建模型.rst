使用 TensorFlow 和 PyTorch 创建模型
===============================================

`Jupyter Notebook <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Creating_Models_with_TensorFlow_and_PyTorch.ipynb>`_

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/%E4%BD%BF%E7%94%A8%20TensorFlow%20%E5%92%8C%20PyTorch%20%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8B.ipynb>`_

`Jupyter Notebook 中文翻译版下载 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/%E4%BD%BF%E7%94%A8%20TensorFlow%20%E5%92%8C%20PyTorch%20%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8B.ipynb>`_

在之前的教程中，我们使用的是 DeepChem 提供的标准模型。这对于许多应用来说都没问题，但是迟早你会希望使用你自己定义的体系结构创建一个全新的模型。DeepChem提供了与 TensorFlow (Keras) 和 PyTorch 的集成，所以你可以在这两个框架的模型中使用它。

实际上，在DeepChem中使用TensorFlow或PyTorch模型时，你可以采用两种不同的方法。这取决于你是想使用TensorFlow/PyTorch APIs 还是DeepChem APIs 来训练和评估你的模型。对于前一种情况，DeepChem的 `Dataset` 类有一些方法可以方便地将其与其他框架一起使用。 `make_tf_dataset()` 返回一个遍历数据的 `tensorflow.data.Dataset` 对象。 `make_pytorch_dataset()` 返回一个遍历数据的 `torch.utils.data.IterableDataset` 对象。这让你可以使用 DeepChem 的数据集（datasets）、加载器（loaders）、特征器（featurizers）、变换器（transformers）、拆分器（splitters）等，并轻松将它们集成到你现有的 TensorFlow 或 PyTorch 代码中。

但 DeepChem 还提供了许多其他有用的功能。另一种让你使用这些功能的方法是将你的模型包装在一个 DeepChem 的 `Model` 对象中。让我们看看如何做到这一点。

KerasModel
------------------

`KerasModel` 是DeepChem `Model` 类的子类。它充当 `tensorflow.keras.Model` 的包装器。让我们看一个使用它的例子。对于本例，我们创建了一个由两个密集层组成的简单顺序（sequential）模型。

.. code-block:: Python

    import deepchem as dc
    import tensorflow as tf

    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1)
    ])
    model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())

对于本例，我们使用了Keras的 `Sequential` 类。我们的模型由一个具有 ReLU 激活的密集层、50% 的 dropout 提供正则化和最后一个产生标量输出的层组成。我们还需要指定在训练模型时使用的损失函数，在本例中 :math:`L^2` 损失。我们现在可以训练和评估该模型，就像我们对任何其他 DeepChem 模型一样。例如，让我们加载 Delaney 溶解度数据集。我们的模型如何基于分子的扩展连通性指纹 (ECFPs) 来预测分子的溶解性?

.. code-block:: Python

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    model.fit(train_dataset, nb_epoch=50)
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))

TorchModel
-------------------

`TorchModel` 的工作原理与 `KerasModel` 类似，只不过它包装了一个 `torch.nn.Module`。让我们使用PyTorch来创建另一个模型，就像前面的模型一样，并用相同的数据训练它。

.. code-block:: Python

    import torch

    pytorch_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1000, 1)
    )
    model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())

    model.fit(train_dataset, nb_epoch=50)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))

损失的计算
-----------

现在让我们看一个更高级的例子。在上述模型中，损失是直接从模型的输出计算出来的。这通常是可以的，但并非总是如此。考虑一个输出概率分布的分类模型。虽然从概率中计算损失是可能的，但从 logits 中计算损失在数值上更稳定。

为此，我们创建一个返回多个输出的模型，包括概率和 logits。 `KerasModel` 和 `TorchModel` 让你指定一个“输出类型（output_types）”列表。如果一个特定的输出具有 `'prediction'` 类型，这意味着它是一个正常的输出，在调用 `predict()` 时应该返回。如果它有 `'loss'` 类型，这意味着它应该传递给损失函数，而不是正常的输出。

顺序模型不允许多个输出，因此我们使用子类化样式模型（subclassing style model）。

.. code-block:: Python

    class ClassificationModel(tf.keras.Model):
        
        def __init__(self):
            super(ClassificationModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(1000, activation='relu')
            self.dense2 = tf.keras.layers.Dense(1)

        def call(self, inputs, training=False):
            y = self.dense1(inputs)
            if training:
                y = tf.nn.dropout(y, 0.5)
            logits = self.dense2(y)
            output = tf.nn.sigmoid(logits)
            return output, logits

    keras_model = ClassificationModel()
    output_types = ['prediction', 'loss']
    model = dc.models.KerasModel(keras_model, dc.models.losses.SigmoidCrossEntropy(), output_types=output_types)

我们可以在 BACE 数据集中训练我们的模型。这是一个二元分类任务，试图预测一个分子是否会抑制BACE-1酶。

.. code-block:: Python

    tasks, datasets, transformers = dc.molnet.load_bace_classification(feturizer='ECFP', splitter='scaffold')
    train_dataset, valid_dataset, test_dataset = datasets
    model.fit(train_dataset, nb_epoch=100)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))

类似地，我们将创建一个自定义的分类器模型类来与 `TorchModel` 一起使用。理由跟与上面的 `KerasModel` 相似，自定义模型允许轻松得到第二个密集层的未缩放输出(Tensorflow 中的 logits)。自定义类允许定义如何向前传递；在最终的sigmoid被应用产生预测之前得到 logits。

最后，用一个需要概率和 logits 的 `ClassificationModel` 的实例与一个损失函数搭配生成一个 `TorchModel` 的实例进行训练。

.. code-block:: Python

    class ClassificationModel(torch.nn.Module):
        
        def __init__(self):
            super(ClassificationModel, self).__init__()
            self.dense1 = torch.nn.Linear(1024, 1000)
            self.dense2 = torch.nn.Linear(1000, 1)

        def forward(self, inputs):
            y = torch.nn.functional.relu( self.dense1(inputs) )
            y = torch.nn.functional.dropout(y, p=0.5, training=self.training)
            logits = self.dense2(y)
            output = torch.sigmoid(logits)
            return output, logits

    torch_model = ClassificationModel()
    output_types = ['prediction', 'loss']
    model = dc.models.TorchModel(torch_model, dc.models.losses.SigmoidCrossEntropy(), output_types=output_types)

我们将使用相同的 BACE 数据集。和以前一样，该模型将尝试进行二元分类任务，试图预测一个分子是否会抑制BACE-1酶。

.. code-block:: Python

    tasks, datasets, transformers = dc.molnet.load_bace_classification(feturizer='ECFP', splitter='scaffold')
    train_dataset, valid_dataset, test_dataset = datasets
    model.fit(train_dataset, nb_epoch=100)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    print('training set score:', model.evaluate(train_dataset, [metric]))
    print('test set score:', model.evaluate(test_dataset, [metric]))

其他功能
---------

`KerasModel` 和 `TorchModel` 有很多其他的功能。下面列一些比较重要的。

- 训练过程中自动保存检查点（checkpoints）。
- 将进度记录到控制台（console）或者传送到 `TensorBoard <https://www.tensorflow.org/tensorboard>`_ 或 `Weights & Biases <https://docs.wandb.com/>`_ 。
- 以 `f(输出，标签，权重)` 的形式定义损失函数。
- 使用 `ValidationCallback` 类来提前停止。
- 从已训练的模型加载参数。
- 估计模型输出的不确定性。
- 通过显著性映射（saliency mapping）识别重要特征。

通过将你自己的模型包装在 `KerasModel` 或 `TorchModel` 中，你就可以使用所有这些功能。有关它们的详细信息，请参阅API文档。
