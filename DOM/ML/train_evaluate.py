from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    """
    训练模型、进行预测并评估模型性能。

    参数:
        classifier: 分类器对象（如 SVM）。
        train_features: 训练集特征（如 TF-IDF 特征）。
        train_labels: 训练集标签。
        test_features: 测试集特征（如 TF-IDF 特征）。
        test_labels: 测试集标签。

    返回:
        y_pred: 测试集的预测结果。
    """
    # 1. 训练模型
    classifier.fit(train_features, train_labels)

    # 2. 进行预测
    y_pred = classifier.predict(test_features)

    # 3. 评估模型
    # 计算准确率
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    # # 计算 F1 得分
    # f1 = f1_score(test_labels, y_pred, average='weighted')
    # print(f"F1 得分: {f1:.2f}")
    #
    # # 计算召回率
    # recall = recall_score(test_labels, y_pred, average='weighted')
    # print(f"召回率: {recall:.2f}")
    #
    # # 打印分类报告
    # print("分类报告:")
    # print(classification_report(test_labels, y_pred))
    #
    # # 打印混淆矩阵
    # print("混淆矩阵:")
    # print(confusion_matrix(test_labels, y_pred))

    return y_pred