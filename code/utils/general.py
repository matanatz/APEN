import os

import logging




def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_item(list,idx):
    if (len(list) > 0):
        return list[idx]
    else:
        return None



def write_to_clipboard(output):
    import subprocess
    process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode())
        
def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def configure_logging(debug,quiet,logfile):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("equiv_shapes - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)

