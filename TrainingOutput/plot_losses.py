import matplotlib.pyplot as plt


with open("training_losses.txt", "r") as f:
    data = f.read().split("\n")[:-1]  # Ignore last line, it's (usually) empty

print(f"There are {len(data)} losses")


data = [d.split(",") for d in data]  # Split by comma, as it's csv
data = [float(d[2]) for d in data]  # Take the combined loss, though feel free to check out other two values

smoothed = []
step = 10  # Averaging out every ten losses into one to get a smoother interpretation
for i in range(0, len(data), step):
    smoothed.append(sum(data[i:i+step])/step)


plt.figure(1)
plt.plot(data, color="black")

plt.figure(2)
plt.plot(smoothed, color="black")

plt.show()



data = data[-1000:]  # To check the training of last 1000 losses
smoothed = []
step = 10  # Averaging out every ten losses into one to get a smoother interpretation
for i in range(0, len(data), step):
    smoothed.append(sum(data[i:i+step])/step)



plt.figure(1)
plt.plot(data, color="black")

plt.figure(2)
plt.plot(smoothed, color="black")

plt.show()

