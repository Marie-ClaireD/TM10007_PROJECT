

# The training set is again divided into a training and a validation set and afterwards classified using a leave one out validation and logistic regression.

def leave_one_out_val(x,y):
    """
    Leave One Out Cross Validation using Logistic Regression as a classifier
    """
    loo = LeaveOneOut()
    loo.get_n_splits(x,y)

    predict_labels = [] 
    predict_proba =[]
    y_val_total = []

    for train_index, val_index in loo.split(x,y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val= y[train_index], y[val_index]

        if min(x_train.shape[0], x_train.shape[1]) < 70:
            print('Not enough input values for PCA with 70 components')
            sys.exit()
        else:
            pca = PCA(n_components=70)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)

            lrg= LogisticRegression()
            lrg.fit(x_train,y_train) 
    
            lrg_predicted=lrg.predict(x_val)
            predict_labels.append(lrg_predicted)
            predict = lrg.predict_proba(x_val)[:,1]
            predict_proba.append(predict)
            y_val_total.append(y_val)
    predict_labels = np.array(predict_labels)
    predic_proba = np.array(predict_proba)

    return predict_labels, predict_proba, y_val_total

predict_labels_loo, predict_proba_loo, y_val_total_loo = leave_one_out_val(x_train,y_train)

# The training set is again divided into a training and a validation set and afterwards classified using a Kfold cross validation and logistic regression.
def cross_val(x,y):
    """
    Cross validation using a Logistic Regression classifier (5 folds)
    """

    crss_val = RepeatedKFold(n_splits = 5, n_repeats=10, random_state = None)           
    crss_val.get_n_splits(x, y)

    predict_labels = []
    predict_probas = []
    y_val_total = []

    #idx = np.arange(0, len(y))

    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val= y[train_index], y[val_index]

        if min(x_train.shape[0], x_train.shape[1]) < 70:
            print('Not enough input values for PCA with 70 components')
            sys.exit()
        else:
            pca = PCA(n_components=70)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)

            lrg=LogisticRegression()
            lrg.fit(x_train,y_train) 
            prediction=lrg.predict(x_val)
            predict_labels.append(prediction)
            predict = lrg.predict_proba(x_val)[:,1]
            predict_probas.append(predict)
            y_val_total.append(y_val)

    predict_labels = np.array(predict_labels)
    predict_probas = np.array(predict_probas)
    return predict_labels, predict_probas, y_val_total

predict_labels_crss, predict_proba_crss, y_val_total_crss = cross_val(x_train, y_train)



def plot_roc(y_test, y_score, n_components):
    """
    Plots ROC curves
    Function returns none
    """
    try:
        # Make dictionaries
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Create plots
        plt.figure()
        lw_ = 1
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw_, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw_, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic, {n_components}\
                  principle components')
        plt.legend(loc="lower right")
        plt.show(block=False)

        # Catch an unexpected error
    except Exception as error:
        print('Unexpected error:'+str(error))
        raise
    return None


def compute_performance(labels_test, predict_test):
    """
    Compute accuracy, sensitivity and specificity based on confusion matrix.
    """
    try:
        # Form confusion matrix
        conf_matrix = confusion_matrix(labels_test, predict_test).ravel()

        # Calculate accuracy
        accuracy = (conf_matrix[0]+conf_matrix[3])/(conf_matrix[0] +
                                                    conf_matrix[1] + conf_matrix[2]
                                                    + conf_matrix[3])
        # Calculate sensitivity
        sensitivity = conf_matrix[3]/(conf_matrix[3]+conf_matrix[2])

        # Calculate specificity
        specificity = conf_matrix[0]/(conf_matrix[0]
                                      +conf_matrix[1])

    except Exception as error:
        print("Unexpected error:"+str(error))
        raise
    
    return(accuracy, sensitivity, specificity)




performance_clf = []
clsfs = [LogisticRegression(), KNeighborsClassifier(n_neighbors=hyperparameters[0].get('n_neighbors')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]
clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

for clf in clsfs:
    performances = leave_one_out_pca(x_train, y_train, hyperparameters, clf) 
    performance_clf.append(performances)

print(performance_clf)