def jaccard_coefficient_binary(a, b):
    intersection = torch.sum(a & b)
    union = torch.sum(a | b)
    return intersection.float() / union.float