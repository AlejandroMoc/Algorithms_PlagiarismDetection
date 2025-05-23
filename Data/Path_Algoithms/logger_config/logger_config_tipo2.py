import logging

def configure_logging():
    logr = logging.getLogger('Path_Algorithms')
    logr.setLevel(logging.DEBUG)
    hndlr = logging.StreamHandler()
    frmttr = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hndlr.setFormatter(frmttr)
    logr.addHandler(hndlr)
    return logr

logr = configure_logging()