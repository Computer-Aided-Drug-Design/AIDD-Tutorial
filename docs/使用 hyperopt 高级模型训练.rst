使用 Hyperopt 高级模型训练
==================================================================

`Jupyter Notebook <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Advanced_model_training_using_hyperopt.ipynb>`_

`Jupyter Notebook 中文翻译版查看 <https://github.com/abdusemiabduweli/AIDD-Tutorial-Files/blob/main/DeepChem%20Jupyter%20Notebooks/使用%20hyperopt%20高级模型训练.ipynb>`_

`Jupyter Notebook 中文翻译版下载 <https://abdusemiabduweli.github.io/AIDD-Tutorial-Files/DeepChem%20Jupyter%20Notebooks/使用%20hyperopt%20高级模型训练.ipynb>`_

在高级模型训练教程中，我们已经了解了在 deepchem 包中使用 GridHyperparamOpt 进行超参数优化。在本教程中，我们将研究另一个称为 Hyperopt 的超参数调优库。

准备
------

要运行本教程需要安装 Hyperopt 库。

.. code-block:: Python

    pip install hyperopt

通过hyperopt进行超参数优化
-------------------------------

让我们从加载 HIV 数据集开始。它根据是否抑制艾滋病毒复制对超过4万个分子进行了分类。

.. code-block:: Python

    import deepchem as dc
    tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
    train_dataset, valid_dataset, test_dataset = datasets

现在，让我们导入 hyperopt 库，我们将使用它来提供最佳参数

.. code-block:: Python

    from hyperopt import hp, fmin, tpe, Trials

然后，我们必须声明一个字典，其中包含所有超形参及其将调优的范围。这本字典将作为 hyperopt 的搜索空间。

在字典中声明范围的一些基本方法是：

*   hp.choice('label',[*choices*]) : this is used to specify a list of choices
*   hp.uniform('label' ,low= **low_value** ,high= **high_value** ) :  this is used to specify a uniform distibution

在低值和高值之间。它们之间的值可以是任何实数，不一定是整数。

在这里，我们将使用多任务分类器对 HIV 数据集进行分类，因此适当的搜索空间如下所示。

.. code-block:: Python

    search_space = {
        'layer_sizes': hp.choice('layer_sizes',[[500], [1000], [2000],[1000,1000]]),
        'dropouts': hp.uniform('dropout',low=0.2, high=0.5),
        'learning_rate': hp.uniform('learning_rate',high=0.001, low=0.0001)
    }

然后，我们应该声明一个由 hyperopt 最小化的函数。所以，这里我们应该使用这个函数来最小化我们的多任务分类器模型。此外，我们使用 validation callback 在每1000步验证分类器，然后将最佳分数作为返回值传递。这里使用的指标是 ` roc_auc_score ` ，需要最大化它。使一个非负值最大化相当于使其负数最小化，因此我们将返回验证得分的负数。

.. code-block:: Python

    import tempfile
    #tempfile is used to save the best checkpoint later in the program.

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    def fm(args):
    save_dir = tempfile.mkdtemp()
    model = dc.models.MultitaskClassifier(n_tasks=len(tasks),n_features=1024,layer_sizes=args['layer_sizes'],dropouts=args['dropouts'],learning_rate=args['learning_rate'])
    #validation callback that saves the best checkpoint, i.e the one with the maximum score.
    validation=dc.models.ValidationCallback(valid_dataset, 1000, [metric],save_dir=save_dir,transformers=transformers,save_on_minimum=False)
    
    model.fit(train_dataset, nb_epoch=25,callbacks=validation)

    #restoring the best checkpoint and passing the negative of its validation score to be minimized.
    model.restore(model_dir=save_dir)
    valid_score = model.evaluate(valid_dataset, [metric], transformers)

    return -1*valid_score['roc_auc_score']

在这里，我们调用 hyperopt 的 fmin 函数，在这里我们传递要最小化的函数、要遵循的算法、最大 eval 数和一个 trials 对象。Trials 对象用于保存所有超参数、损失和其他信息，这意味着你可以在运行优化后访问它们。此外，Trials 可以帮助你保存重要信息，以便稍后加载，然后恢复优化过程。

此外，该算法有三种选择，无需额外配置即可使用。他们是:-


*   Random Search - rand.suggest
*   TPE (Tree Parzen Estimators) - tpe.suggest
*   Adaptive TPE - atpe.suggest

下面的代码用于打印 hyperopt 找到的最佳超参数。

.. code-block:: Python

    print("Best: {}".format(best))

这里发现的超参数不一定是最好的，但可以大致了解哪些参数是有效的。为了得到更准确的结果，必须增加验证周期的数量和模型拟合的周期。但是这样做可能会增加寻找最佳超参数的时间。