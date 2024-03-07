def HSVToRGB(HSV):

    [h, s, v] = HSV

    if s == 0.0:
        return v, v, v
    i = int(h*6.)  # Assume H is given as a value between 0 and 1.
    f = (h*6.)-i
    p, q, t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f))
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q