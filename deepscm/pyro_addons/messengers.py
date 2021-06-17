import collections
import functools

from pyro.poutine.messenger import Messenger


class SiteScaleMessenger(Messenger):
    def __init__(self, scale_dict):
        super().__init__()
        self.scale_dict = scale_dict

    def _process_message(self, msg):
        if msg['name'] in self.scale_dict:
            msg['scale'] *= self.scale_dict[msg['name']]
        return None


def site_scale_messenger(fn=None, *args, **kwargs):
    if fn is not None and not (callable(fn) or isinstance(fn, collections.abc.Iterable)):
        raise ValueError(
            "{} is not callable, did you mean to pass it as a keyword arg?".format(fn))
    msngr = SiteScaleMessenger(*args, **kwargs)
    return functools.update_wrapper(msngr(fn), fn, updated=()) if fn is not None else msngr
