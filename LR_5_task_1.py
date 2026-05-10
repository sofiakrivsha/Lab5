import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# Функція візуалізації 
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
def run_task(classifier_type='rf'):
    # Вказуємо прямий шлях до файлу, щоб не було помилок
    input_file = r'C:\Users\sofia\Downloads\data_random_forests.txt'
    if not os.path.exists(input_file):
        print(f"Файл не знайдено за шляхом: {input_file}")
        return
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        print("\n" + "="*30 + "\nRANDOM FOREST\n" + "="*30)
        classifier = RandomForestClassifier(**params)
    else:
        print("\n" + "="*30 + "\nEXTRA TREES\n" + "="*30)
        classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_test, y_test, f'Test dataset ({classifier_type})')
    y_test_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_test_pred))
if __name__ == '__main__':
    run_task(classifier_type='rf')
    run_task(classifier_type='erf')
    plt.show()
