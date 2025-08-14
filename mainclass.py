"""                      SPAM MAIL DETECTION USING SUPPORT VECTOR MACHINE (SVM)                    """

# Importing the modules
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tkinter import Tk, Button, Label, Text
from sklearn.decomposition import PCA


class SpamMailDetection :

    """
     This class is responsible for reading, preprocessing, vectorizing the data and then data is
     feed into SVM Classifier, and it also contains a GUI part to interact with Ml algorithm for
     prediction.
    """

    def __init__(self, path):

        """ Initialization function. It is responsible for initialization Tf-idf Vectorization,
        SVM Classifier and the DataFrame.
        :param path -> it takes filename or path of the dataset as input
        """

        # Reading the given Excel file
        self.excelFile = pd.read_excel(path)

        # Loading the file into a Data Frame
        self.data = pd.DataFrame(self.excelFile, columns = ['v1', 'v2'])

        # Defining Text Vectorization method td-idf
        self.vectorizer = TfidfVectorizer()

        # Defining the ML model SVM with the parameters
        self.classifier = svm.SVC(kernel = 'linear',
                                  gamma = 'auto', C = 2,
                                  verbose = True)

        # Test dataset initialization
        self.Y_test = None
        self.train_feature = None
        self.test_feature = None

    def preprocessing(self):

        """ Pre-processing the loaded data, like checking for NA values, converting the labels into
        integers, splitting the dataset for both training and testing purposes.
        :return -> returns the split data into train X, test X, train labels, and test labels
        """

        # Converting the labels into integers
        y_vals = np.array([1 if y == 'spam' else 0 for y in self.data.v1])
        # print(y_vals[:50])

        # Splitting the dataset into training and testing purpose
        X_train, X_test, Y_train, self.Y_test = train_test_split(self.data.v2, y_vals,
                                                                    test_size = 0.2,
                                                                    random_state = 42)

        # Converting the inputs explicitly to strings
        X_train_str = [str(text) for text in X_train]
        X_test_str = [str(text) for text in X_test]

        return X_train_str, Y_train, X_test_str, self.Y_test

    def textVectorization(self, X_train, X_test):

        """ It takes to array of strings, and uses Tf-idf technique of vectorization for converting
         the strings into numeric vales as SVM doesn't take strings as input. And returns the
         vectorized form of the input array of strings.
         :return -> vectorized form input array of strings
         """

        # Text Vectorization with tf-idf technique
        self.train_feature = self.vectorizer.fit_transform(X_train)
        self.test_feature = self.vectorizer.transform(X_test)

        # Printing information about the Text Vectorization with tf-idf technique
        """ print('\nidf values:')
        for ele1, ele2 in zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_):
            print(ele1, ':', ele2)

        # get indexing
        print('\nWord indexes:')
        print(self.vectorizer.vocabulary_)

        # display tf-idf values
        print('\ntf-idf value:')
        print(train_features) """

        # Save the tf-idf vectorizer
        vectorizername = "tfidf_vectorizer.pkl"
        pickle.dump(self.vectorizer.vocabulary_, open(vectorizername, "wb"))

        # Save the features
        np.save("train_features", self.train_feature)
        np.save("test_features", self.test_feature)

        return self.train_feature, self.test_feature

    def train(self, X, y):

        """ This function fit the parameter into the classifier, save the trained model and return
        the classifier.
         :param X -> input data
         :param y -> labels
         :return trained model/classifier
         """

        # Training dataset
        self.classifier.fit(X, y)

        # Save the model to disk
        modelname = 'svm_model.sav'
        pickle.dump(self.classifier, open(modelname, 'wb'))

        return self.classifier

    def load_model(self):

        """ It loads all the weights of the trained model, vectorizer or vocabulary of the vectorizer
         for further use it in program with creating and training the model once again. """

        # Load the tf-idf vectorizer
        vectorizername = "tfidf_vectorizer.pkl"
        self.vectorizer = CountVectorizer(decode_error = 'replace', vocabulary = pickle.load(open(vectorizername, "rb")))

        # Load the already trained model from disk
        modelname = 'svm_model.sav'
        self.classifier = pickle.load(open(modelname, 'rb'))

    def predict(self, X_t, flagV = False):

        """ The function takes an input and tries to find out the predicted class for that string.
        :param X_t -> input string
        :param flagV -> True if the input string is already vectorized, otherwise false
        :return -> returns the predicted value if the input is an array of strings, otherwise returns
                a string.
        """

        # Checking if the input is Vectorized, if yes then direct goes for prediction,
        # otherwise first get vectorized and then goes for prediction
        # flagV = True, if X_t is already Vectorized
        # flagV = False, if X_t is not Vectorized
        if not flagV: # False
            feature = self.vectorizer.transform(X_t)
            y_predict = self.classifier.predict(feature)
        else:
            # Predicting the test dataset on the trained model
            y_predict = self.classifier.predict(X_t)

        if len(y_predict) == 1:
            # Checking the category of the prediction
            if y_predict == 1:
                return "Spam!!"
            elif y_predict == 0:
                return "Not a Spam email!"
            else:
                print("ERROR!!!")
                return y_predict
        else:
            return y_predict

    def report(self):

        """ This function calculate the accuracy score and generate a report. It also creates all
        the necessary graphs, and save it in the directory called graphs. """

        # Predicting the output for the test cases
        Y_predict = self.predict(self.test_feature, flagV = True)

        # Calculating the accuracy
        accuracy = accuracy_score(self.Y_test, Y_predict)
        print("Accuracy:", accuracy * 100)

        # Report
        print("Classification Report :\n", classification_report(self.Y_test, Y_predict))


        # Creating the Confusion matrix
        confusion_matrix = metrics.confusion_matrix(self.Y_test, Y_predict)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,
                                                    display_labels = [False, True])
        cm_display.plot()
        plt.savefig("./graphs/Confusion Matrix.png")
        plt.show()


        # Creating SVM Hyperplane
        # Perform PCA to reduce dimensions for visualization
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(self.train_feature.toarray())

        # Create a meshgrid for visualization
        x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
        y_min, y_max = X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Project the meshgrid points back to the original feature space
        meshgrid_in_original_space = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

        # Get the decision function for each point in the projected meshgrid
        Z = self.classifier.decision_function(meshgrid_in_original_space)

        # Reshape the result to the meshgrid shape
        Z = Z.reshape(xx.shape)

        # Create a contour plot for the SVM decision boundary
        plt.contourf(xx, yy, Z, levels = [-1, 0, 1], alpha = 0.5, cmap = plt.cm.RdYlBu)

        # Scatter plot for support vectors
        plt.scatter(X_pca[self.classifier.support_, 0], X_pca[self.classifier.support_, 1], c = 'red',
                    marker = 'x', label='Support Vectors')

        # Set labels and title
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("SVM Hyperplane and Support Vectors (PCA Visualization)")

        # Show legend
        plt.legend()
        plt.savefig("./graphs/SVM Hyperplane.png")

        plt.show()

    def guiHandling(self):

        """ This function is responsible for all the elements in the GUI interface, it takes the input
        string and predict the output with the respective function, and returns it the interface. """

        # Create the main window
        window = Tk()
        window.geometry("600x400+290+110")
        window.title("  EMAIL SPAM DETECTOR  ")
        window.configure(bg = "#D4E1E3")

        # Create the heading
        heading = Label(window, text = "  EMAIL SPAM DETECTOR  ", bg = "#D4E1E3", fg = "black", bd = 0,
                        font = ('Bahnschrift SemiLight', 18))
        heading.place(x = 170, y = 7)

        # Create the text box
        text_box = Text(window, bg = "white", fg = "black", bd = 1, height = 16, width = 70,
                        font = ('Bahnschrift SemiLight', 10), cursor = "xterm")
        text_box.place(x = 50, y = 100)

        # Create the label
        label = Label(window, bg = "#D4E1E3", fg = "red", bd = 1,
                        font = ('Bahnschrift SemiLight', 15))
        label.place(x = 350, y = 55)

        # Bind the button click event to a function
        def click_btn():
            text = text_box.get('1.0', 'end')
            output = self.predict([str(text)], flagV = False)
            print(output)
            label.config(text = output)

        # Create the button
        button = Button(window, text = "   CHECK   ", fg = "black", bd = 0, height = 2, width = 20,
                        font = ('Bahnschrift SemiLight', 10), cursor = "arrow", command = click_btn)
        button.place(x = 50, y = 50)

        # Start the main loop
        window.mainloop()


if __name__ == "__main__":

    # Calling the Class
    filename: str = "./datasets/Spam Email Detection.xlsx"
    smd = SpamMailDetection(filename)
    # x_train, y_train, x_test, y_test = smd.preprocessing()
    # train_features, test_features = smd.textVectorization(x_train, x_test)
    # smd.train(train_features, y_train)
    # smd.report()
    smd.load_model()
    smd.guiHandling()

