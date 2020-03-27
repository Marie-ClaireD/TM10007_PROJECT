# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

  # Hyperparameters: 
    # n_estimators: number of decision trees
    # bootstrap = True 
    # max_depth: default is none, so the decision trees can be prone to overfitting. --> other value has to be given. 
    # min_samples_leaf: specifies the minimum number of samples required to be at a leaf node. The default value for this parameter is 1, which means that every leaf      must have at least 1 sample that it classifies.
    # random_state = 0: to obtain a deterministic behaviour during fitting

    # Vb. Exhaustive Grid search:
    # n_estimators = [100, 300, 500, 800, 1200]
    # max_depth = [5, 8, 15, 25, 30]
    # min_samples_split = [2, 5, 10, 15, 100]
    # min_samples_leaf = [1, 2, 5, 10] 

    # hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
                  # min_samples_split = min_samples_split, 
                  # min_samples_leaf = min_samples_leaf)

    # gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, n_jobs = -1)
    # bestF = gridF.fit(x_train, y_train)

homemade_random_forest = BaggingClassifier(DecisionTreeClassifier())
voting_ensemble = VotingClassifier(
    estimators=[('KNN', KNeighborsClassifier()), ('tree', DecisionTreeClassifier()), ('rf', RandomForestClassifier())],
    voting='soft')
clsfs = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
         homemade_random_forest, voting_ensemble]

def cross_val_clf(x,y):
    """
    Cross validation using a Random Forest classifier (5 folds)
    """

    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.5)            
    ss.get_n_splits(x, y)

    performances = []
    list_performances = [] 
    for clf in clsfs:
        for train_index, val_index in ss.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val= y[train_index], y[val_index]

            #rf=RandomForestClassifier()
            clf.fit(x_train,y_train) 
            prediction=clf.predict(x_val)
            accuracy = accuracy_score(y_val, prediction)
            performances.append(accuracy)

    return performances

accuracy = cross_val_clf(x_train, y_train)
print(accuracy)

def cross_val_rf(x,y):
    """
    Cross validation using a Random Forest classifier (5 folds)
    """

    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.5)            
    ss.get_n_splits(x, y)

    n_trees = [1, 5, 10, 50, 100]

    performances = []
    for n_tree in n_trees:
        clf = RandomForestClassifier(n_estimators=n_tree, bootstrap=True, random_state=0)
        for train_index, val_index in ss.split(x, y):
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val= y[train_index], y[val_index]

                clf.fit(x_train,y_train) 
                prediction=clf.predict(x_val)
                accuracy = accuracy_score(y_val, prediction)
                performances.append(accuracy)

        return performances

accuracy = cross_val_rf(x_train, y_train)
print(accuracy)

def leave_one_out_val_clf(x,y):
    """
    Leave One Out Cross Validation using Random Forest as a classifier
    """

    loo = LeaveOneOut()
    loo.get_n_splits(x,y)

    LeaveOneOut() 

    prediction = [] 
    y_val_total = []
    performance = []
    for clf in clsfs:
        for train_index, val_index in loo.split(x,y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val= y[train_index], y[val_index]
        
            clf.fit(x_train,y_train) 
            clf_predicted=clf.predict(x_val)
            prediction.append(clf_predicted)
            y_val_total.append(y_val)
            accuracy = accuracy_score(y_val_total, prediction)
            performance.append(accuracy)

    return performance

accuracy = leave_one_out_val_clf(x_train,y_train)
print(accuracy)

def leave_one_out_val_rf(x,y):
    """
    Leave One Out Cross Validation using Random Forest as a classifier
    """

    loo = LeaveOneOut()
    loo.get_n_splits(x,y)

    LeaveOneOut() 

    n_trees = [1, 5, 10, 50, 100]

    prediction = [] 
    y_val_total = []
    performance = []

    for n_tree in n_trees:
        clf = RandomForestClassifier(n_estimators=n_tree, bootstrap=True, random_state=0)
        for train_index, val_index in loo.split(x,y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val= y[train_index], y[val_index]
        
            clf.fit(x_train,y_train) 
            clf_predicted=clf.predict(x_val)
            prediction.append(clf_predicted)
            y_val_total.append(y_val)
        accuracy = accuracy_score(y_val_total, prediction)
        performance.append(accuracy)

    return performance

accuracy = leave_one_out_val_rf(x_train,y_train)
print(accuracy)