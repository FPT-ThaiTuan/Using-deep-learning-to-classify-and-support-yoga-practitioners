from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
from demo import *

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

data_train = data_train.drop(labels="Unnamed: 0", axis=1)  # delete if error about it
X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
model = SVC(kernel='poly', decision_function_shape='ovo')
model.fit(X, Y)
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=True, min_detection_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

# Test phase : build test dataset then evaluate
# data_test.drop(labels="Unnamed: 0", axis=1, inplace=True)  # delete if error about it
# predictions = evaluate(data_test, model, show=True)
# I think downdog and plank get these results because of the absence of variations in the hands positions
# Unlike tree and goddess which present some hand position variations

# Create a confusion matrix
# cm = confusion_matrix(data_test['target'], predictions)

# # Display the confusion matrix using a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

correct_feedback(model,'test2.mp4')
#predict_video(model, "vid2.mp4", show=True)
#cv2.destroyAllWindows()
