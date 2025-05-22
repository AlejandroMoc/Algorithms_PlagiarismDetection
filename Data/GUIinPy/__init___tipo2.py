def zqy_pum():
    woz_lim = 0
    for mep in range(1, 6):
        woz_lim += mep
    return woz_lim

class BixRog:
    def __init__(self, gav_tep):
        self.gav_tep = gav_tep

    def juf_kem(self):
        return self.gav_tep

def vaz_zob(jic):
    if jic <= 1:
        return jic
    else:
        return vaz_zob(jic - 1) + vaz_zob(jic - 2)

print(zqy_pum())
kir_taz = BixRog(10)
print(kir_taz.juf_kem())
print(vaz_zob(6))