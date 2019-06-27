import sys
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score,precision_recall_curve,recall_score,average_precision_score,roc_curve,roc_auc_score
from sklearn.metrics import f1_score,precision_recall_fscore_support,classification_report,log_loss
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
from inspect import signature
#style.use('ggplot')

'''
# Save all the Print Statements in a Log file.
old_stdout = sys.stdout
log_file = open("summary.log","w")
sys.stdout = log_file
'''
# Load MNIST Data
#print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')

#print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

#print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

#Features
X = train_img

#Labels
y = train_labels

# Prepare Classifier Training and Testing Data
#print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)

# Pickle the Classifier for Future Use
#print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
#print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train,y_train)

with open('MNIST_SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle','rb')
clf = pickle.load(pickle_in)

#print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

#print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

#print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)

#print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test,y_pred)

print('SVM Trained Classifier Accuracy: ',acc)
#print('\nPredicted Values: ',y_pred)
print('Accuracy of Classifier on Validation Images: ',accuracy)
print('Accuracy percentage on validation images',accuracy*100,'%')
#print('\nConfusion Matrix: \n',conf_mat)

#print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

#print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
acc = accuracy_score(test_labels,test_labels_pred)

#print('\nCalculating precision of Trained Classifier on Test Data... ')
precision = precision_score(test_labels,test_labels_pred)

#print('\nCalculating recall of Trained Classifier on Test Data... ')
recall = recall_score(test_labels,test_labels_pred)

#print('\nCalculating average precision of Trained Classifier on Test Data... ')
avg_precision = average_precision_score(test_labels,test_labels_pred)

#print('\nCalculating roc auc score of Trained Classifier on Test Data... ')
roc_auc_score = roc_auc_score(test_labels,test_labels_pred)

#print('\n Creating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

f1_score = f1_score(test_labels,test_labels_pred)

support = precision_recall_fscore_support(test_labels,test_labels_pred)

report = classification_report(test_labels,test_labels_pred)

loss = log_loss(test_labels,test_labels_pred)

#print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('Accuracy of Classifier on Test Images: ',acc)
print('Accuracy percentage:',acc*100,'%')
print('precision',precision)
print('recall',recall)
print('average precision',avg_precision)
print('roc auc score',roc_auc_score)
print('f1 score',f1_score)
print('support',support)
print('report',report)
print('loss',loss)

precision, recall, _ = precision_recall_curve(test_labels,test_labels_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          avg_precision))
plt.show()


fpr, tpr, thresh = roc_curve(test_labels,test_labels_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of using SVM model')
plt.legend()
plt.show()