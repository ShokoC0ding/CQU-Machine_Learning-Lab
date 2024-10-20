from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


def draw_cart_bySklearn():
    iris = datasets.load_iris()
    X, y = datasets.load_iris(return_X_y=True)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("dotGraph_code1")

draw_cart_bySklearn()
