from k_fold import *
from splitData import *

# data extraction
data = []
with open('temp_pre.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X, Y = [], []
        X.append(int(row["temperature"]))
        Y.append(float(row["pressure"]))
        data.append([X, Y])

# validation using 1:5 ratio
train, valid = split(data)
print("training set:\n",train)
print("validation set\n", valid)
print()

with open("split.txt", "w") as output:
    output.write("training set:\n")
    for e in train:
        output.write(str(e))
        output.write("\n")
    output.write("validation set:\n")
    for e in valid:
        output.write(str(e))
        output.write("\n")

# cross-validation using k-folds
part = k_fold(data)
for e in part:
    print(e)

with open("k_folds.txt", "w") as output:
    for e in part:
        output.write(str(e))
        output.write("\n")