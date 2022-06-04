import time
from datetime import datetime, timedelta
from collections.abc import Iterable

DATEFORMAT = {'tweet': '%a %b %d %H:%M:%S %z %Y',
              'youtube': '%Y-%m-%d'}

# twitter's snowflake parameters
twepoch = 1288834974657
datacenter_id_bits = 5
worker_id_bits = 5
sequence_id_bits = 12
max_datacenter_id = 1 << datacenter_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (64 - datacenter_id_bits - worker_id_bits - sequence_id_bits)


def melt_snowflake(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its components."""
    snowflake_id = int(snowflake_id)
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    datacenter_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_datacenter_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> datacenter_id_bits
    timestamp_ms += twepoch
    return timestamp_ms, datacenter_id, worker_id, sequence_id


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print('>>> Elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - self.start_time))[:-3]))


def str2obj(str, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return datetime.strptime(str, DATEFORMAT[fmt])
    else:
        return datetime.strptime(str, fmt)


def obj2str(obj, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return obj.strftime(DATEFORMAT[fmt])
    else:
        return obj.strftime(fmt)


def concise_fmt(x, pos):
    if abs(x) // 10000000000 > 0:
        return '{0:.0f}B'.format(x / 1000000000)
    elif abs(x) // 1000000000 > 0:
        return '{0:.1f}B'.format(x / 1000000000)
    elif abs(x) // 10000000 > 0:
        return '{0:.0f}M'.format(x / 1000000)
    elif abs(x) // 1000000 > 0:
        return '{0:.0f}M'.format(x / 1000000)
    elif abs(x) // 10000 > 0:
        return '{0:.0f}K'.format(x / 1000)
    elif abs(x) // 1000 > 0:
        return '{0:.0f}K'.format(x / 1000)
    else:
        return '{0:.0f}'.format(x)


def hide_spines(axes):
    if isinstance(axes, Iterable):
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
