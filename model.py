from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def get_classifier(name):
    if name == "SVM":
        return SVC(kernel="rbf", gamma="scale", C=1)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100)
    else:
        return GaussianNB()

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, preds
