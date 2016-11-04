"""Handy set of simulation functions

-----
"""
import pandas as pd
import numpy as np


def estimate_frequency(tidx):
    """calculate the esitmated numeric frequency of time series

    :param tidx: series of dates
    :type tidx: DatetimeIndex

    :rtype: float

    Examples

    >>> import pandas as pd
    >>> import numpy as np
    >>> import simutil
    >>> tidx = pd.date_range('2008-01-01', '2019-12-31', freq='B')
    >>> util.estimate_frequency(tidx)
    261.0

    """
    tidx = pd.to_datetime(tidx)
    first = tidx.min()
    last = tidx.max()
    days_in_range = (last - first).days
    num_periods = tidx.shape[0] - 1
    days_in_year = 365.25
    return round(days_in_year * num_periods / days_in_range, 0)


def randn_returns(securities, tidx, annual_return=.04, annual_volatility=.12):
    """Simulate log normal returns.
    First row will be all zeros..

    Assumptions

    - Annual expected return of `4%`
    - Annual volatility of `12%`
    - Returns are log normal

    .. math:: \\ln{(1 + r)} \\sim N(\\mu, \\sigma)

    With:

    .. math:: E(annual\_return) = 4\%
    .. math:: f = annual\_frequency

    we can estimate :math:`\\mu`

    .. math:: \\mu = \\frac{\\ln{(1 + E(annual\_return))}}{f}

    and estimate :math:`\\sigma`

    .. math:: \\sigma = \\frac{12\%}{\\sqrt{f}}

    :param securities: list of security Id's
    :type securities: list

    :param tidx: series of dates
    :type tidx: DatetimeIndex

    :param annual_return: expected annual return
    :type annual_return: float

    :param annual_volatility: expected annual volatility
    :type annual_volatility: float

    :rtype: DataFrame

    Examples

    >>> import numpy as np
    >>> import simutil
    >>> np.random.seed([3,1415])
    >>> securities = list('AB')
    >>> tidx = ['2011-12-31', '2012-06-30',
    >>>         '2012-12-31', '2013-06-30',
    >>>         '2013-12-31', '2014-06-30']
    >>> simutil.randn_returns(list('AB'), tidx)
    Id                 A         B
    2011-12-31  0.000000  0.000000
    2012-06-30 -0.148795 -0.084260
    2012-12-31 -0.137217 -0.158086
    2013-06-30 -0.009977  0.017474
    2013-12-31  0.047539  0.050436
    2014-06-30  0.083624  0.088730

    """
    tidx = pd.to_datetime(tidx)
    securities = pd.Index(securities, name='Id')
    freq = estimate_frequency(tidx)
    mu = np.log(1 + annual_return) / freq
    sigma = annual_volatility / np.sqrt(freq)
    a = np.random.lognormal(mean=mu, sigma=sigma,
                            size=(tidx.shape[0], securities.shape[0]))
    return pd.DataFrame(a - 1, tidx, securities, dtype=np.float64)


def implied_prices(tot_rets, price, dividends):
    """We can back out what price must be to arrive at total returns.
    We model total returns with a log normal distrubution.  Therefore, it
    becomes important to be able to deduce what prices would be given a
    dividend.

    We will use some clever array algorithms to quickly calculate the
    implied prices

    :param tot_rets: DataFrame of total returns in decimal space.
        Index contains the DatetimeIndex and Columns are security ids
    :type tot_rets: DataFrame

    :param price: Series of initial prices indexed by security ids
    :type price: Series

    :param dividends: Series of static dividends
        Index values are security ids
    :type dividends: Series

    :rtype: DataFrame
    """

    # perform reverse cumprod on total returns
    T = tot_rets.add(1).sort_index(ascending=False).cumprod().sort_index()
    # combine initial prices with dividends
    # price is positive in the first row
    # while all dividends are negative
    D = pd.DataFrame(np.zeros_like(T), T.index, T.columns).sub(dividends)
    D.iloc[0] += price
    prices = T.mul(D).cumsum().div(T.shift(-1).fillna(1))
    prices['_CSH'] = 1.

    return prices


def prices_asof(prices, date):
    """Return the row from the prices DataFrame with the most
    recently available date as of the date provided.

    :param prices: time series of all prices
    :type prices: DataFrame

    :param date: The date to get prices as of
    :type date: Timestamp

    :rtype: Series
    """
    date = pd.to_datetime(date)
    return prices.iloc[prices.index.get_loc(date, 'ffill')]


def to_shares(weights, prices, nominal_value, date, floor=0):
    """Convert a Series of weights to shares

    :param weights: Series of weights with Ids in the index
    :type weights: Series

    :param prices: Time series of reference prices
    :type prices: DataFrame

    :param nominal_value: Assumed market value of the set of holdings
    :type nominal_value: float

    :param date: Date to reference prices
    :type date: Timestamp

    :param floor: Number of decimals to round down.  Positive means left of decimal place
    :type floor: int

    :rtype: Series
    """
    p = prices_asof(prices, date)
    v = weights.mul(nominal_value)
    s = v.div(p).dropna()
    e = 10 ** floor
    return (s // e) * e


def random_allocation(prices, frac=.5, freq='A', precision=2):
    """Use the columns from the prices reference to produce
    random allocations to the specified securities.

    :param prices: Time series of reference prices
    :type prices: DataFrame

    :param frac: Probability of non-zero allocation
    :type frac: float

    :param freq: pandas frequency string for resampling
    :type freq: str

    :param precision: Number of decimals to round weights
    :type precision: float

    :rtype: DataFrame
    """
    allocation_df = pd.concat(
        [prices, prices.index.to_series().rename('Date')],
        axis=1
    ).resample(freq).first().set_index('Date')

    allocation_df.loc[:] = np.random.rand(*allocation_df.shape)
    allocation_df = allocation_df.mask(
        np.random.choice([True, False], allocation_df.shape, p=[1 - frac, frac]))
    if '_CSH' in allocation_df.columns:
        allocation_df = allocation_df.drop('_CSH', 1)

    allocation_df = allocation_df.fillna(0).div(allocation_df.sum(1).add(1e-2), 0).round(precision)
    last_col = allocation_df.columns[-1]
    allocation_df.loc[:, last_col] -= allocation_df.sum(1).sub(1)
    return allocation_df


def allocations_asof(allocations, date):
    """Return the row from the allocations DataFrame with the most
    recently available date as of the date provided.

    :param allocations: time series of all allocations
    :type allocations: DataFrame

    :param date: The date to get prices as of
    :type date: Timestamp

    :rtype: Series
    """
    date = pd.to_datetime(date)
    return allocations.iloc[allocations.index.get_loc(date, 'ffill')]


