class CSVUtils:
    def __init__(self, fn):
        self.fn = fn

        with open(self.fn, "r") as f:
            header = f.readline().strip().split(",")

        self.header_pos = {}
        for i, key in enumerate(header):
            self.header_pos[key] = i

    def query(self, matches, queries):
        match = ",".join(matches)

        with open(self.fn, "r") as f:
            line = f.readline()
            while line[:len(match)] != match:
                line = f.readline()

                if line == "":
                    raise ValueError("Bad match")

        split = line.strip().split(",")

        query_dict = {}
        for query in queries:
            i = self.header_pos[query]
            val = self.trans_val(split[i])

            query_dict[query] = val

        return query_dict

    def get_all(self):
        data = []
        with open(self.fn, "r") as f:
            line = f.readline()
            while line:
                split = line.strip().split(",")
                data.append([self.trans_val(val) for val in split])

                line = f.readline()

        return data


    def get_prefixes(self, num_cols):
        prefixes = []
        with open(self.fn, "r") as f:
            # Skip the header
            f.readline()

            line = f.readline()
            while line:
                prefix = [self.trans_val(val)
                          for val in line.split(",")[:num_cols]]
                prefixes.append(prefix)
                line = f.readline()

        return prefixes

    def trans_val(self, val):
        if val.isdigit():
            val = int(val)
        elif val.replace(".", "", 1).replace("e", "", 1).replace("+", "", 1).replace("-", "", 1).isdigit():
            val = float(val)

        return val
