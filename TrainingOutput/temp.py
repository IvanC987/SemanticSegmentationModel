
import matplotlib.pyplot as plt


with open("temp1.txt", "r") as f:
    scaled = f.read().split("\n")

print(len(scaled))


with open("temp2.txt", "r") as f:
    nscaled = f.read().split("\n")

print(len(nscaled))


with open("ce.txt", "r") as f:
    ce = f.read().split("\n")

print(len(ce))


with open("de.txt", "r") as f:
    de = f.read().split("\n")

print(len(de))

de = de[500:]




scaled = [e.split(",") for e in scaled]
nscaled = [e.split(",") for e in nscaled]
ce = [e.split(",") for e in ce]
de = [e.split(",") for e in de]


temp = []
for i in range(0, len(de), 10):
    temp.append(sum([float(t[1]) for t in de[i:i+10]])/10)

# plt.plot([float(i[1]) for i in scaled], c="r")
# print([i[0] for i in scaled])
# # exit()
# plt.plot([float(i[1]) for i in nscaled], c="b")
plt.plot(temp, c="g")
plt.show()










