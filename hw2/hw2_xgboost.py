import numpy as np
import pandas as pd
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

x = pd.read_csv(sys.argv[1],encoding='big5')
y = pd.read_csv(sys.argv[2],encoding='big5')

x = np.array(x)
y = np.squeeze(y)


seed = 7
test_size = 1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

x_test = pd.read_csv(sys.argv[3],encoding='big5')
x_test = np.array(x_test)

result = model.predict(x_test)

f = open(sys.argv[4], "w")

f.write("id,label\n")

for i in range(len(result)):
	f.write(str(i+1)+","+str(result[i])+"\n")

f.close()
