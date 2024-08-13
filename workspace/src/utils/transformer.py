from src.utils.csv_utils import CSVUtils


class Transformer:
    def __init__(self, B, D, E, F, H, M, P, S):
        self.B = B
        self.D = D
        self.E = E
        self.F = F
        self.H = H
        self.M = M
        self.P = P
        self.S = S

    @classmethod
    def from_csv(cls, *matches):
        csv = CSVUtils("../outputs/pregenerated/results/flat_validation.csv")
        ranks = csv.query(matches, ["B", "D", "E", "F", "H", "M", "P", "S"])

        return cls(**ranks)

    def partition(self, rank, rank1, rank0, rank0_shape):
        shape = getattr(self, rank)
        setattr(self, rank1, shape // rank0_shape)
        setattr(self, rank0, rank0_shape)

    def update_problem(self, yaml_dict):
        instance = yaml_dict["problem"]["instance"]
        for rank in instance.keys():
            instance[rank] = getattr(self, rank)

    def __key(self):
        return (self.B, self.D, self.E, self.F, self.H, self.M, self.P, self.S)

    def __eq__(self, other):
        if not isinstance(other, Transformer):
            return False

        return self.__key() == other.__key()
