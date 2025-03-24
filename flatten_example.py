import numpy as np

featureMaps = [
    [[3, 4, 7], [5, 6, 1], [1, 8, 12]],
    [[6, 12, 10], [0, 21, 13], [17, 9, 19]],
]

featureMaps = np.array(featureMaps)

print("****************** INPUT TO FLATTEN LAYER ******************")

for i, featureMap in enumerate(featureMaps):
    print("\nFeature Map " + str(i + 1) + " : \n" + str(featureMap))

flattenOutput = featureMaps.flatten()

print("\n\n****************** OUTPUT OF FLATTEN LAYER ******************")

print("\nFlattened Layer Output: \n" + str(flattenOutput), end="\n\n")
