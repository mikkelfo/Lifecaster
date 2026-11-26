import bisect

ONE_YEAR_ABSPOS = 365.25 * 24


def find_censor_idx(abspos, censor, key=None):
    """Finds censoring idx"""
    if isinstance(abspos, list):
        key = lambda x: x[0]
    last_valid_idx = bisect.bisect_left(abspos, censor, key=key)
    if last_valid_idx == 0:
        raise ValueError("Censoring before background would result in empty sequences")
    return last_valid_idx


def calc_age_from_abspos(first_abspos, target_abspos):
    """Calculates the amount of years between first_abspos and target_abspos"""
    return (target_abspos - first_abspos) / ONE_YEAR_ABSPOS
