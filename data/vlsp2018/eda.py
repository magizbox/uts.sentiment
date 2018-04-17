import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, dirname
from pylab import rcParams


def analyze(path, name):
    df = pd.read_excel(path, encoding='sys.getfilesystemencoding()')
    print("Dataset '{}' is loaded!".format(name))
    print("\t- size:", df.shape)
    rcParams['figure.figsize'] = 30, 15
    df.drop("text", axis=1).sum().plot.barh()
    plt.savefig(join(output_path, "{}_labels_distribution.png".format(name)))
    plt.clf()


output_path = join(dirname(__file__), "eda")
train_hotel = join(dirname(__file__), "corpus", "train", "hotel.xlsx")
train_resto = join(dirname(__file__), "corpus", "train", "restaurant.xlsx")
test_hotel = join(dirname(__file__), "corpus", "test", "hotel.xlsx")
test_resto = join(dirname(__file__), "corpus", "test", "restaurant.xlsx")

analyze(train_hotel, "hotel_train")
analyze(train_resto, "restaurant_train")
analyze(test_hotel, "hotel_test")
analyze(test_resto, "restaurant_tesst")
