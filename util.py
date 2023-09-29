def split_on_window(sequence, limit, step):
    ret = []
    split_sequence = sequence.split()
    l, r = 0, limit
    while r < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
        l += step
        r += step
    if l < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
    return ret
