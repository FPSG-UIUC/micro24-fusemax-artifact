# Read the timeloop-model.stats.txt
class Stats:
    def __init__(self, fn):
        self.fn = fn

    def read_data(self, prefixes):
        # Read a piece of data by finding the first instance of each
        # prefix sequentially and returning the data associated with the
        # last line

        with open(self.fn, "r") as f:
            line = f.readline().strip()

            for prefix in prefixes:
                while line[:len(prefix)] != prefix:
                    line = f.readline()

                    if line == "":
                        break

                    line = line.strip()

        i = line.index(":")

        return self.__format_line(line[i + 2:])

    def __format_line(self, line):
        if line.isdigit():
            return int(line)
        else:
            return line
