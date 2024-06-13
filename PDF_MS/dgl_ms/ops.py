from mindspore import ops


def edge_softmax(graph, bases):
    new_bases = None
    sum_bases = ops.sum(bases, dim=0, keepdim=True)
    for eattr in bases:
        if new_bases == None:
            new_bases = eattr / sum_bases
        else:
            new_bases = ops.cat(new_bases, eattr / sum_bases)
    return new_bases