import os

base_path = "dataset/train"

real_count = len(os.listdir(os.path.join(base_path, "real")))
fake_count = len(os.listdir(os.path.join(base_path, "fake")))

print("Real images:", real_count)
print("Fake images:", fake_count)
