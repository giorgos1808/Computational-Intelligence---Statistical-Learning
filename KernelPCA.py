# --KernelPCA--
# Tsakiris Giorgos

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import KernelPCA
import time
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.datasets import mnist, cifar10
import seaborn as sns
import matplotlib.pyplot as plt

# Project 2
def KPCA_LDA(dataset='mnist', examples=False, kpca_params=[None,  'linear', None, 3],
        knn_params=[True, 5, 'uniform', 'minkowski'], cn_params=[True, 'euclidean']):

    if dataset == 'mnist':
        (X, y), (_, _) = mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

    X = np.reshape(X, (X.shape[0], -1))
    y = np.reshape(y, (-1,))

    # normalize X -> [0,1] from [0,255]
    X = np.float32(X) / 255
    y = np.float32(y)

    with open('results_kpca_lda_' + dataset + '.txt', 'a') as f:
        print('KPCA parameters: n_components= %s, kernel= %s, gamma= %s, degree= %d' % (str(kpca_params[0]), kpca_params[1], str(kpca_params[2]), kpca_params[3]))
        f.write('KPCA parameters: n_components= %s, kernel= %s, gamma= %s, degree= %d\n' % (str(kpca_params[0]), kpca_params[1], str(kpca_params[2]), kpca_params[3]))
        print('Nearest Neighbor parameters: n_neighbors= %d, weights= %s, metric= %s' % (knn_params[1], knn_params[2], knn_params[3]))
        f.write('Nearest Neighbor parameters: n_neighbors= %d, weights= %s, metric= %s\n' % (knn_params[1], knn_params[2], knn_params[3]))
        print('Nearest Class Centroid parameters: metric= %s' % cn_params[1])
        f.write('Nearest Class Centroid parameters: metric= %s\n' % cn_params[1])

        # Split
        # 30000 examples -> training set = 18000 , test set = 12000
        X, _, y, _ = train_test_split(X, y, test_size=0.5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        start_kpca = time.time()
        kpca_model = KernelPCA(n_components=kpca_params[0], kernel=kpca_params[1], gamma=kpca_params[2], degree=kpca_params[3])
        X_data_features = X.shape[1]
        X_train_kpca = kpca_model.fit_transform(X_train)
        X_test_kpca = kpca_model.transform(X_test)
        end_kpca = time.time()
        print('KPCA time: %0.2f sec' % (end_kpca - start_kpca))
        f.write('KPCA time: %0.2f sec\n' % (end_kpca - start_kpca))
        print('X features: %d , X_kpca features: %d' % (X_data_features, X_train_kpca.shape[1]))
        f.write('X features: %d , X_kpca features: %d\n' % (X_data_features, X_train_kpca.shape[1]))

        # lda for dimensionality reduction. It should keep [classes-1] components
        start_lda = time.time()
        lda_model = LDA()
        X_data_features = X_train_kpca.shape[1]
        X_train_kpca_lda = lda_model.fit_transform(X_train_kpca, y_train)
        X_test_kpca_lda = lda_model.transform(X_test_kpca)
        end_lda = time.time()
        print('LDA time: %0.2f sec' % (end_lda - start_lda))
        f.write('LDA time: %0.2f sec\n' % (end_lda - start_lda))
        print('X_kpca features: %d , X_kpca_lda features: %d' % (X_data_features, X_train_kpca_lda.shape[1]))
        f.write('X_kpca features: %d , X_kpca_lda features: %d\n' % (X_data_features, X_train_kpca_lda.shape[1]))

        # Nearest Neighbor
        if knn_params[0]:
            start_training = time.time()
            knc_model = KNeighborsClassifier(n_neighbors=knn_params[1], weights=knn_params[2], metric=knn_params[3])
            knc_model.fit(X_train_kpca_lda, y_train)
            end_training = time.time()
            start_predict = time.time()
            y_train_predict = knc_model.predict(X_train_kpca_lda)
            y_test_predict = knc_model.predict(X_test_kpca_lda)
            end_predict = time.time()
            print('Nearest Neighbor')
            f.write('Nearest Neighbor\n')
            print('Training Accuracy: %0.4f' % accuracy_score(y_train, y_train_predict))
            f.write('Training Accuracy: %0.4f\n' % accuracy_score(y_train, y_train_predict))
            print('Test Accuracy: %0.4f' % accuracy_score(y_test, y_test_predict))
            f.write('Test Accuracy: %0.4f\n' % accuracy_score(y_test, y_test_predict))
            print('Training time: %0.2f sec' % (end_training - start_training))
            f.write('Training time: %0.2f sec\n' % (end_training - start_training))
            print('Prediction time: %0.2f sec' % (end_predict - start_predict))
            f.write('Prediction time: %0.2f sec\n' % (end_predict - start_predict))

            if examples:
                confusionMatrix = confusion_matrix(y_test, y_test_predict)
                sns.heatmap(confusionMatrix.T, annot=True, fmt='d', cbar=False)
                plt.title("Confusion matrix of Nearest Neighbor on %s test set" % dataset)
                plt.xlabel('True output')
                plt.ylabel('Predicted output')
                plt.show()

        # Nearest Class Centroid
        if cn_params[0]:
            start_training = time.time()
            nc_model = NearestCentroid(metric=cn_params[1])
            nc_model.fit(X_train_kpca_lda, y_train)
            end_training = time.time()
            start_predict = time.time()
            y_train_predict = nc_model.predict(X_train_kpca_lda)
            y_test_predict = nc_model.predict(X_test_kpca_lda)
            end_predict = time.time()
            print('Nearest Class Centroid')
            f.write('Nearest Class Centroid\n')
            print('Training Accuracy: %0.4f' % accuracy_score(y_train, y_train_predict))
            f.write('Training Accuracy: %0.4f\n' % accuracy_score(y_train, y_train_predict))
            print('Test Accuracy: %0.4f' % accuracy_score(y_test, y_test_predict))
            f.write('Test Accuracy: %0.4f\n' % accuracy_score(y_test, y_test_predict))
            print('Training time: %0.2f sec' % (end_training - start_training))
            f.write('Training time: %0.2f sec\n' % (end_training - start_training))
            print('Prediction time: %0.2f sec' % (end_predict - start_predict))
            f.write('Prediction time: %0.2f sec\n' % (end_predict - start_predict))

            if examples:
                confusionMatrix = confusion_matrix(y_test, y_test_predict)
                sns.heatmap(confusionMatrix.T, annot=True, fmt='d', cbar=False)
                plt.title("Confusion matrix of Nearest Class Centroid on %s test set" % dataset)
                plt.xlabel('True output')
                plt.ylabel('Predicted output')
                plt.show()


if __name__ == '__main__':
    # Project 2
    # n_components=int, default=None(If None, all non-zero components are kept), kernel{'linear', 'poly', 'rbf', 'sigmoid', 'cosine'},
    # gamma=float (default=None = 1/n_features), degree=int (degree of 'poly')
    # kpca_params=[None,  'linear', None, 3] #len = 4

    # if call knn, n_neighbors int, weights {'uniform', 'distance'}, metric  {'minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean'}
    # Knn_params = [True, 5, 'uniform', 'minkowski'] #len = 4

    # if call nc, metric  {'euclidean', 'manhattan'}
    # Cn_params = [True, 'euclidean'] #len = 2

    KPCA_LDA(dataset='mnist', kpca_params=[106, 'poly', None, 5], knn_params=[True, 5, 'distance', 'cosine'], cn_params=[True, 'euclidean'], examples=True)
    KPCA_LDA(dataset='cifar10', kpca_params=[131, 'rbf', None, 3], knn_params=[True, 5, 'distance', 'minkowski'], cn_params=[True, 'euclidean'], examples=True)
