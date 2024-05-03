from classifier.kmean import Indexer
from classifier.gui import Table


def main():
    indexer = Indexer()
    indexer.read_file("bbcsport/", "files/")

    tb = Table()
    tb.create_Gui(indexer)


if __name__ == "__main__":
    main()
