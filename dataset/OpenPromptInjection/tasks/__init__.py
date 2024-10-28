from .TargetTask import TargetTask
from .InjectedTask import InjectedTask
from .TargetTaskICL import TargetTaskICL
from .InjectedTaskICL import InjectedTaskICL

def create_task(config, data_num, icl_num=20, for_injection=False):
    """
    Factory function to create the task object
    """
    if not for_injection:
        return TargetTask(config, data_num, icl_num)
    else:
        return InjectedTask(config, data_num, icl_num)


def create_icl_task(config, data_num, icl_num=20, for_injection=False):
    """
    Factory function to create the task object
    """
    if not for_injection:
        return TargetTaskICL(config, data_num, icl_num)
    else:
        return InjectedTaskICL(config, data_num, icl_num)