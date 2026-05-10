import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
# Вбудована функція візуалізації
def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), 
                                 np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.title(title)
if __name__ == '__main__':
    input_file = r'C:\Users\sofia\Downloads\data_random_forests.txt'
    if not os.path.exists(input_file):
        print(f"Файл не знайдено: {input_file}")
        exit()

    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    # Визначення сітки параметрів для перевірки 
    parameter_grid = [
        {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
        {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
    ]
    metrics = ['precision_weighted', 'recall_weighted']
    for metric in metrics:
        print("\n" + "#"*60)
        print(f"#### Searching optimal parameters for {metric}")
        print("#"*60)
        # Створення та запуск GridSearchCV 
        classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric
        )
        classifier.fit(X_train, y_train)
        print("\nGrid scores for the parameter grid:")
        # Виведення результатів для кожної комбінації
        for params, avg_score in zip(classifier.cv_results_['params'], 
                                     classifier.cv_results_['mean_test_score']):
            print(f"{params} --> {round(avg_score, 3)}")
        print(f"\nBest parameters: {classifier.best_params_}")
        y_pred = classifier.predict(X_test)
        print("\nPerformance report:\n")
        print(classification_report(y_test, y_pred))
        visualize_classifier(classifier, X_test, y_test, f"Best for {metric}")
    plt.show()
