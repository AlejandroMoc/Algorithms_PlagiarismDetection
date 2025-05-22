def zorblify(blep):
    gloop = 0
    for snarf in blep:
        if snarf % 2 == 0:
            gloop += snarf
    return gloop


def flibberize(flabber):
    glip = []
    for blork in flabber:
        glip.append(blork * 2)
    return glip


class Zazzle:
    def __init__(self, zibble):
        self.zibble = zibble

    def zorp(self):
        return self.zibble * 3