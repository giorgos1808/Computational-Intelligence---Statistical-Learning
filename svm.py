# --SVM--
# Tsakiris Giorgos

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import time
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from keras.datasets import mnist, cifar10
import seaborn as sns
import matplotlib.pyplot as plt

# Project 1
def SVM(dataset='mnist', pca=True, examples=False, svm_params=[True, 1.0, 'rbf', 3, 'scale'],
        knn_params=[True, 5, 'uniform', 'minkowski'], cn_params=[True, 'euclidean']):

    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X = np.reshape(X, (X.shape[0], -1))
    y = np.reshape(y, (-1,))

    # normalize X -> [0,1] from [0,255]
    X = np.float32(X) / 255
    y = np.float32(y)

    with open('results_'+dataset+'.txt', 'a') as f:
        print('SVM parameters: C= %f, kernel= %s, degree= %d, gamma= %s' %(svm_params[1], svm_params[2], svm_params[3], svm_params[4]))
        f.write('SVM parameters: C= %f, kernel= %s, degree= %d, gamma= %s\n' %(svm_params[1], svm_params[2], svm_params[3], svm_params[4]))
        print('Nearest Neighbor parameters: n_neighbors= %d, weights= %s, metric= %s' %(knn_params[1], knn_params[2], knn_params[3]))
        f.write('Nearest Neighbor parameters: n_neighbors= %d, weights= %s, metric= %s\n' %(knn_params[1], knn_params[2], knn_params[3]))
        print('Nearest Class Centroid parameters: metric= %s' % cn_params[1])
        f.write('Nearest Class Centroid parameters: metric= %s\n' % cn_params[1])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        # PCA
        if pca:
            start_pca = time.time()
            pca_model = PCA(0.92)
            X_data_features = X.shape[1]
            X_train = pca_model.fit_transform(X_train)
            X_test = pca_model.transform(X_test)
            end_pca = time.time()
            print('PCA time: %0.2f sec' %(end_pca - start_pca))
            f.write('PCA time: %0.2f sec\n' % (end_pca - start_pca))
            print('X features: %d , X_pca features: %d' %(X_data_features, X_train.shape[1]))
            f.write('X features: %d , X_pca features: %d\n' %(X_data_features, X_train.shape[1]))

        # SVM
        if svm_params[0]:
            start_training = time.time()
            SVC_model = SVC(C=svm_params[1], kernel=svm_params[2], degree=svm_params[3], gamma=svm_params[4])
            SVC_model.fit(X_train, y_train)
            end_training = time.time()
            start_predict = time.time()
            y_train_predict = SVC_model.predict(X_train)
            y_test_predict = SVC_model.predict(X_test)
            end_predict = time.time()
            print('SVM')
            f.write('SVM\n')
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
                plt.title("Confusion matrix of Support Vector Machine on %s test set" % dataset)
                plt.xlabel('True output')
                plt.ylabel('Predicted output')
                plt.show()

        # Nearest Neighbor
        if knn_params[0]:
            start_training = time.time()
            knc_model = KNeighborsClassifier(n_neighbors=knn_params[1], weights=knn_params[2], metric=knn_params[3])
            knc_model.fit(X_train, y_train)
            end_training = time.time()
            start_predict = time.time()
            y_train_predict = knc_model.predict(X_train)
            y_test_predict = knc_model.predict(X_test)
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
            nc_model.fit(X_train, y_train)
            end_training = time.time()
            start_predict = time.time()
            y_train_predict = nc_model.predict(X_train)
            y_test_predict = nc_model.predict(X_test)
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
    # Project 1
    # if call svm, C=float (Regularization parameter), kernel {'linear', 'poly', 'rbf', 'sigmoid'}, degree (degree of 'poly'), gamma {'scale', 'auto'}
    # Svm_params = [True, 1.0, 'rbf', 3, 'scale']    #len = 5

    # if call knn, n_neighbors int, weights {'uniform', 'distance'}, metric  {'minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean'}
    # Knn_params = [True, 5, 'uniform', 'minkowski'] #len = 4

    # if call nc, metric  {'euclidean', 'manhattan'}
    # Cn_params = [True, 'euclidean'] #len = 2

    SVM(dataset='mnist', svm_params=[True, 5.0, 'poly', 3, 'scale'], knn_params=[True, 5, 'distance', 'l2'], cn_params=[True, 'euclidean'], examples=True)
    SVM(dataset='cifar10', svm_params=[True, 10.0, 'rbf', 3, 'auto'], knn_params=[True, 5, 'distance', 'cosine'], cn_params=[True, 'manhattan'], examples=True)
