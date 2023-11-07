import numpy as np
from sklearn.svm import SVC
from collections import Counter



def ensemble_of_classifiers(projected_train, train_y):
    """
    For each category a classifier is trained to discriminate between this category and all the other categories put together.

    Parameters
    ----------
    projected_train : shape (num_samples_train, projection_dim)
    train_y : shape (num_samples_train)


    Returns
    -------
    classifiers_list : a list of trained classifiers. The it'h classifier desined to predict the i'th category.
    diagnosis_list : list of the categories order as the classifiers.
    """

    classifiers_list = []

    diagnosis_list = np.sort(list(Counter(list(train_y)).keys()))  # orderd by increasing order: (0,1,2,..)

    for diagnosis in diagnosis_list:
        print(diagnosis)

        y = np.zeros(len(train_y))
        is_diagnosis = train_y == diagnosis
        y[is_diagnosis] = 1
        y = y.astype('int')
        if y.sum() == 0:  # TODO delete
            continue
        classifier = SVC(gamma='auto', probability=True)
        classifier.fit(projected_train, y)

        classifiers_list.append(classifier)

    return classifiers_list, diagnosis_list

